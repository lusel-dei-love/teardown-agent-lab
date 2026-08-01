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
local TOWER_DIST = 4.0 -- metres in front of the player spawn
local JITTER_M = 0.05  -- per-episode initial-state randomisation

-- Host command channel: the host creates/removes these files in the mod folder.
local RESET_FILE = "MOD/reset.txt"

-- Registry keys. The engine rewrites these under savegame.mod.local-<modfolder>.
local STATE_KEY = "savegame.mod.state"
local READY_KEY = "savegame.mod.ready"

--------------------------------------------------------------------------------
-- Mutable state
--------------------------------------------------------------------------------
local blocks = {}       -- entity handles, ordered
local spawns = {}       -- Vec, spawn position per block (same order)
local player_spawn = nil
local tower_origin = nil
local episode = 0
local seq = 0
local episode_t0 = 0
local reset_seen = false

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

-- Build the tower: COLS wide, ROWS high, centred on tower_origin.
local function build_tower()
	SetRandomSeed(episode)
	spawns = {}
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
		spawns[idx] = pos
	end
	refresh_handles()
end

local function reset_episode()
	episode = episode + 1
	clear_blocks()
	build_tower()
	if player_spawn then
		SetPlayerTransform(player_spawn)
	end
	episode_t0 = GetTime()
end

--------------------------------------------------------------------------------
-- State serialisation
--
-- seq|t|episode|seed|px,py,pz|yaw,pitch|b0x,b0y,b0z;b1x,...  then spawn poses.
-- Kept to digits, '.', '-', ',', ';', '|' so rapidxml never sees a metacharacter.
--------------------------------------------------------------------------------
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
	}

	local cur = {}
	local spn = {}
	local loc = {}
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

	SetString(STATE_KEY, table.concat(parts, "|"))
end

--------------------------------------------------------------------------------
-- Callbacks
--------------------------------------------------------------------------------
function init()
	player_spawn = GetPlayerTransform(0)
	local fwd = TransformToParentVec(player_spawn, Vec(0, 0, -1))
	tower_origin = VecAdd(player_spawn.pos, VecScale(fwd, TOWER_DIST))
	-- Drop the tower base to roughly the player's feet.
	tower_origin[2] = player_spawn.pos[2]

	build_tower()
	episode_t0 = GetTime()
	SetString(READY_KEY, "1")
end

function tick()
	-- Host command channel: reset on the rising edge of the file appearing, so the
	-- host can hold the file until it observes the episode counter advance.
	local want_reset = HasFile(RESET_FILE)
	if want_reset and not reset_seen then
		reset_episode()
	end
	reset_seen = want_reset

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
