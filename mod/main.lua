-- ABOUTME: Teardown-side half of the bridge: spawns the tower-knockdown task, streams
-- ABOUTME: state to the host through the registry, and resets on a host file command.

--------------------------------------------------------------------------------
-- Task layout. Sizes are in voxels; the engine uses 10 voxels per metre.
--------------------------------------------------------------------------------
local BLOCK_VOX = 5 -- 0.5 m cubes
local BLOCK_M = BLOCK_VOX / 10
local COLS = 3
local ROWS = 3
local N_BLOCKS = COLS * ROWS
local TOWER_DIST_MIN = 3.5
local TOWER_DIST_MAX = 6.0
local JITTER_M = 0.05  -- per-episode initial-state randomisation
-- Randomise WHERE the tower is, per episode, seeded by the episode index.
-- With the tower always dead ahead, a constant "walk forward and swing" - no vision at
-- all - scored 80%, matching the trained student, so the task could not tell perception
-- from a reflex. Spawning it on a random bearing forces the agent to actually look for
-- it: the same blind policy then scores 0%.
local TOWER_BEARING_SPREAD = math.pi -- +/- 180 deg around the player's facing

-- Credit a block only if the AGENT STRUCK it. Displacement alone gave a blind
-- "walk forward and swing" 40% - it wandered into the tower and the success rule could
-- not tell a swing from a bump, so the benchmark rewarded accidents. A block counts only
-- if it moved while the player was swinging and within reach of it.
local STRIKE_RANGE = 3.0     -- metres; a sledge cannot reach further
local SWING_MEMORY_TICKS = 20 -- impact can land a moment after the button goes down
local MOVE_EPSILON = 0.02    -- metres per tick; below this is settling, not a hit

-- Host command channel: the host creates/removes these files in the mod folder.
local RESET_FILE = "MOD/reset.txt"
-- Full level reload. The soft reset above respawns blocks but never repairs the world,
-- and this game is destructible: after ~60 episodes of sledge swinging the ground is
-- cratered and the same teacher that solved reliably drops to ~25%. A long collection
-- run must periodically restore the terrain or its later data is measuring a different
-- task than its earlier data.
local HARD_RESET_FILE = "MOD/hardreset.txt"
-- Slow the game clock while a slow model is driving. A VLA needs ~1.4 s per decision
-- while the control loop assumes 10 Hz, so without this the world runs ~14x further
-- between actions than the policy expects and we would be measuring reaction speed
-- rather than decision quality. The factor is reported alongside any result.
local SLOWMO_FILE = "MOD/slowmo.txt"
local SLOWMO_SCALE = 0.1

-- Registry keys. The engine rewrites these under savegame.mod.local-<modfolder>.
local STATE_KEY = "savegame.mod.state"
local READY_KEY = "savegame.mod.ready"

--------------------------------------------------------------------------------
-- Mutable state
--------------------------------------------------------------------------------
local blocks = {}       -- entity handles, ordered
local spawns = {}       -- Vec, settled reference position per block (same order)
local player_spawn = nil
local tower_origin = nil
local episode = 0
local seq = 0
local episode_t0 = 0
local reset_seen = false

-- An episode is only "live" once the freshly built tower has come to rest. Deleting
-- and respawning in the same tick makes the new blocks interpenetrate the old ones and
-- blast apart, which trips the success rule in ~3 s: a false positive that would
-- silently inflate every success rate we measure. So a reset runs as a small state
-- machine and the reference poses are recorded AFTER settling, not at spawn time.
local PHASE_LIVE, PHASE_CLEARING, PHASE_SETTLING = 0, 1, 2
local phase = PHASE_CLEARING
local settle_left = 0
local SETTLE_TICKS = 30

local struck = {}      -- idx -> true once the agent has hit that block
local last_pos = {}    -- idx -> previous position, to detect motion
local swing_ticks = 0  -- counts down from SWING_MEMORY_TICKS while a swing is live

--------------------------------------------------------------------------------
-- Helpers
--------------------------------------------------------------------------------

-- Format a number with fixed precision. Only digits, '.', '-' reach the payload:
-- the engine does NOT escape XML entities, so the payload must stay metacharacter-free.
local function num(v)
	return string.format("%.3f", v)
end

local function vec3(v)
	return num(v[1]) .. "," .. num(v[2]) .. "," .. num(v[3])
end

-- Each block is a tagged dynamic body so it can be found and ordered deterministically
-- after spawning: Spawn()'s return order is not contractual, tag values are.
local function block_xml(idx)
	local s = tostring(BLOCK_VOX)
	return "<body dynamic='true' tags='tdlab_block idx=" .. idx .. "'>"
		.. "<voxbox size='" .. s .. " " .. s .. " " .. s .. "' material='wood'/>"
		.. "</body>"
end

-- Deterministic per-episode jitter: same episode index always yields the same layout.
local function jitter()
	return (GetRandomFloat(0, 1) * 2 - 1) * JITTER_M
end

local function clear_blocks()
	local existing = FindBodies("tdlab_block", true)
	for i = 1, #existing do
		Delete(existing[i])
	end
	blocks = {}
	spawns = {}
end

-- Re-index blocks by their idx tag, so publish order is stable across frames and
-- independent of whatever order Spawn() or FindBodies() happen to return.
local function refresh_handles()
	blocks = {}
	local found = FindBodies("tdlab_block", true)
	for i = 1, #found do
		local idx = tonumber(GetTagValue(found[i], "idx"))
		if idx then
			blocks[idx] = found[i]
		end
	end
end

-- Snapshot where the blocks actually came to rest. Using settled poses (rather than the
-- commanded spawn poses) means normal settling motion never counts as displacement.
local function record_reference_poses()
	spawns = {}
	struck = {}
	last_pos = {}
	for idx = 0, N_BLOCKS - 1 do
		local h = blocks[idx]
		if h then
			spawns[idx] = GetBodyTransform(h).pos
		end
	end
end

-- Where this episode's tower goes: a seeded bearing and distance around the player
-- spawn, so episode N always yields the same layout but successive episodes differ.
local function pick_tower_origin()
	local angle = GetRandomFloat(-TOWER_BEARING_SPREAD, TOWER_BEARING_SPREAD)
	local dist = GetRandomFloat(TOWER_DIST_MIN, TOWER_DIST_MAX)
	local fwd = TransformToParentVec(player_spawn, Vec(0, 0, -1))
	-- Rotate the spawn facing by `angle` about the vertical axis.
	local cos_a, sin_a = math.cos(angle), math.sin(angle)
	local dir = Vec(
		fwd[1] * cos_a - fwd[3] * sin_a,
		0,
		fwd[1] * sin_a + fwd[3] * cos_a
	)
	local origin = VecAdd(player_spawn.pos, VecScale(dir, dist))
	origin[2] = player_spawn.pos[2]
	return origin
end

-- Build the tower: COLS wide, ROWS high, centred on tower_origin.
local function build_tower()
	SetRandomSeed(episode)
	tower_origin = pick_tower_origin()
	for idx = 0, N_BLOCKS - 1 do
		local row = math.floor(idx / COLS)
		local col = idx % COLS
		local offset_x = (col - (COLS - 1) / 2) * BLOCK_M
		local pos = Vec(
			tower_origin[1] + offset_x + jitter(),
			tower_origin[2] + BLOCK_M / 2 + row * BLOCK_M,
			tower_origin[3] + jitter()
		)
		Spawn(block_xml(idx), Transform(pos, QuatEuler(0, 0, 0)))
	end
	refresh_handles()
end

-- Begin a reset. The build happens on a LATER tick so the engine has actually collected
-- the deleted bodies first; see the phase comment above.
local function begin_reset()
	episode = episode + 1
	clear_blocks()
	if player_spawn then
		SetPlayerTransform(player_spawn)
	end
	phase = PHASE_CLEARING
end

local function advance_phase()
	if phase == PHASE_CLEARING then
		build_tower()
		phase = PHASE_SETTLING
		settle_left = SETTLE_TICKS
	elseif phase == PHASE_SETTLING then
		settle_left = settle_left - 1
		if settle_left <= 0 then
			record_reference_poses()
			episode_t0 = GetTime()
			phase = PHASE_LIVE
		end
	end
end

--------------------------------------------------------------------------------
-- State serialisation
--
-- seq|t|episode|seed|px,py,pz|yaw,pitch|b0x,b0y,b0z;b1x,...  then spawn poses.
-- Kept to digits, '.', '-', ',', ';', '|' so rapidxml never sees a metacharacter.
--------------------------------------------------------------------------------
-- Attribute motion to the agent's swing. Runs before publishing so the flags describe
-- the same tick as the positions.
local function update_strikes()
	-- Teardown binds the sledge to the "usetool" action; "lmb" alone missed every swing
	-- and zeroed the teacher as well as the blind policy. Accept either.
	if InputDown("usetool") or InputDown("lmb") then
		swing_ticks = SWING_MEMORY_TICKS
	elseif swing_ticks > 0 then
		swing_ticks = swing_ticks - 1
	end

	local player = GetPlayerPos(0)
	for idx = 0, N_BLOCKS - 1 do
		local h = blocks[idx]
		if h then
			local pos = GetBodyTransform(h).pos
			local previous = last_pos[idx]
			if previous then
				local moved = VecLength(VecSub(pos, previous))
				local reach = VecLength(VecSub(pos, player))
				if moved > MOVE_EPSILON and swing_ticks > 0 and reach < STRIKE_RANGE then
					struck[idx] = true
				end
			end
			last_pos[idx] = pos
		end
	end
end

local function publish_state()
	seq = seq + 1
	local p = GetPlayerPos(0)
	local parts = {
		tostring(seq),
		num(GetTime() - episode_t0),
		tostring(episode),
		tostring(episode), -- seed == episode index (see the transport decision record)
		vec3(p),
		num(GetPlayerYaw(0)) .. "," .. num(GetPlayerPitch(0)),
		tostring(phase), -- 0 live, 1 clearing, 2 settling; only trust state when live
	}

	local cur = {}
	local spn = {}
	local loc = {}
	local hit = {}
	-- Camera space: +x right, +y up, -z forward. Emitting block positions in this frame
	-- means the host never has to guess the engine's yaw convention, and gives any
	-- policy an egocentric observation, which is what it can actually act on.
	local cam = GetPlayerCameraTransform(0)
	for idx = 0, N_BLOCKS - 1 do
		local h = blocks[idx]
		if h and spawns[idx] then
			local t = GetBodyTransform(h)
			cur[#cur + 1] = vec3(t.pos)
			spn[#spn + 1] = vec3(spawns[idx])
			loc[#loc + 1] = vec3(TransformToLocalPoint(cam, t.pos))
		end
	end
	parts[#parts + 1] = table.concat(cur, ";")
	parts[#parts + 1] = table.concat(spn, ";")
	parts[#parts + 1] = table.concat(loc, ";")
	parts[#parts + 1] = table.concat(hit, ";")

	SetString(STATE_KEY, table.concat(parts, "|"))
end

--------------------------------------------------------------------------------
-- Callbacks
--------------------------------------------------------------------------------
function init()
	player_spawn = GetPlayerTransform(0)
	-- tower_origin is chosen per episode in build_tower(), not fixed here.

	-- Episode 0 goes through the same clear/build/settle path as every later reset, so
	-- the reference poses are settled ones in every episode.
	phase = PHASE_CLEARING
	episode_t0 = GetTime()
	SetString(READY_KEY, "1")
end

function tick()
	-- Host command channel: reset on the rising edge of the file appearing, so the
	-- host can hold the file until it observes the episode counter advance.
	if HasFile(HARD_RESET_FILE) then
		Restart()
		return
	end

	SetTimeScale(HasFile(SLOWMO_FILE) and SLOWMO_SCALE or 1.0)

	local want_reset = HasFile(RESET_FILE)
	if want_reset and not reset_seen and phase == PHASE_LIVE then
		begin_reset()
	end
	reset_seen = want_reset

	advance_phase()
	if phase == PHASE_LIVE then
		update_strikes()
	end
	publish_state()
end

function draw()
	UiPush()
	UiTranslate(40, 120)
	UiColor(1, 1, 0)
	UiFont("regular.ttf", 24)
	-- blocks is keyed 0..N-1, so '#' would under-report by one; count explicitly.
	local live = 0
	for idx = 0, N_BLOCKS - 1 do
		if blocks[idx] then
			live = live + 1
		end
	end
	UiText("agent-lab ep=" .. episode .. " blocks=" .. live)
	UiPop()
end
