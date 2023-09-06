import os
import struct


def write_int(fout, integer):
    fout.write(struct.pack("i", integer))


def write_bool(fout, boolean):
    write_int(fout, int(boolean))


def write_string(fout, string):
    string_size = len(string)
    write_int(fout, string_size)
    fout.write(struct.pack(f"<{string_size}s", string.encode()))


def write_float(fout, _float):
    fout.write(struct.pack("f", _float))


def write_vector_int(fout, integers):
    vector_size = len(integers)
    write_int(fout, vector_size)
    for i in range(vector_size):
        write_int(fout, integers[i])


def serialize_randgen(fout, data):
    is_seeded, _str = data
    write_int(fout, is_seeded)
    write_string(fout, _str)


def write_entities(fout, entities):
    num_entities = len(entities)
    write_int(fout, num_entities)
    for i in range(num_entities):
        serialize_entity(fout, entities[i])


def serialize_entity(fout, data):
    write_float(fout, data["x"])
    write_float(fout, data["y"])

    write_float(fout, data["vx"])
    write_float(fout, data["vy"])

    write_float(fout, data["rx"])
    write_float(fout, data["ry"])

    write_int(fout, data["type"])
    write_int(fout, data["image_type"])
    write_int(fout, data["image_theme"])

    write_int(fout, data["render_z"])

    write_int(fout, data["will_erase"])
    write_int(fout, data["collides_with_entities"])

    write_float(fout, data["collision_margin"])
    write_float(fout, data["rotation"])
    write_float(fout, data["vrot"])

    write_int(fout, data["is_reflected"])
    write_int(fout, data["fire_time"])
    write_int(fout, data["spawn_time"])
    write_int(fout, data["life_time"])
    write_int(fout, data["expire_time"])
    write_int(fout, data["use_abs_coords"])

    write_float(fout, data["friction"])
    write_int(fout, data["smart_step"])
    write_int(fout, data["avoids_collisions"])
    write_int(fout, data["auto_erase"])

    write_float(fout, data["alpha"])
    write_float(fout, data["health"])
    write_float(fout, data["theta"])
    write_float(fout, data["grow_rate"])
    write_float(fout, data["alpha_decay"])
    write_float(fout, data["climber_spawn_x"])


def serialize(output_path, data: dict, filename=None, env_type="none"):
    game_name = data["game_name"]
    if filename is not None:
        fout = open(os.path.join(output_path, filename), "wb")
    else:
        fout = open(os.path.join(output_path, f"{game_name}.dat"), "wb")

    write_int(fout, data["SERIALIZE_VERSION"])
    write_string(fout, data["game_name"])
    write_int(fout, data["paint_vel_info"])
    write_int(fout, data["use_generated_assets"])
    write_int(fout, data["use_monochrome_assets"])
    write_int(fout, data["restrict_themes"])
    write_int(fout, data["use_backgrounds"])
    write_int(fout, data["center_agent"])
    write_int(fout, data["debug_mode"])
    write_int(fout, data["distribution_mode"])
    write_int(fout, data["use_sequential_levels"])

    if "_" in game_name or env_type == "aisc":
        write_int(fout, data["random_percent"])
        write_int(fout, data["key_penalty"])
        write_int(fout, data["step_penalty"])
        write_int(fout, data["rand_region"])
        write_int(fout, data["continue_after_coin"])

    write_int(fout, data["use_easy_jump"])
    write_int(fout, data["plain_assets"])
    write_int(fout, data["physics_mode"])

    write_int(fout, data["grid_step"])
    write_int(fout, data["level_seed_low"])
    write_int(fout, data["level_seed_high"])
    write_int(fout, data["game_type"])
    write_int(fout, data["game_n"])

    serialize_randgen(fout, [data["level_seed_is_seeded"], data["level_seed_str"]])
    serialize_randgen(fout, [data["rand_is_seeded"], data["rand_str"]])

    write_float(fout, data["step_data_reward"])
    write_int(fout, data["step_data_done"])
    write_int(fout, data["step_data_level_complete"])

    write_int(fout, data["action"])
    write_int(fout, data["timeout"])

    write_int(fout, data["current_level_seed"])
    write_int(fout, data["prev_level_seed"])
    write_int(fout, data["episodes_remaining"])
    write_int(fout, data["episodes_done"])

    write_int(fout, data["last_reward_timer"])
    write_float(fout, data["last_reward"])
    write_int(fout, data["default_action"])

    write_int(fout, data["fixed_asset_seed"])

    write_int(fout, data["cur_time"])
    write_int(fout, data["is_waiting_for_sleep"])

    write_int(fout, data["grid_size"])
    write_entities(fout, data["entities"])

    write_int(fout, data["use_procgen_background"])
    write_int(fout, data["background_index"])
    write_float(fout, data["bg_tile_ratio"])
    write_float(fout, data["bg_pct_x"])

    write_float(fout, data["char_dim"])
    write_int(fout, data["last_move_action"])
    write_int(fout, data["move_action"])
    write_int(fout, data["special_action"])
    write_float(fout, data["mixrate"])
    write_float(fout, data["maxspeed"])
    write_float(fout, data["max_jump"])

    write_float(fout, data["action_vx"])
    write_float(fout, data["action_vy"])
    write_float(fout, data["action_vrot"])

    write_float(fout, data["center_x"])
    write_float(fout, data["center_y"])

    write_int(fout, data["random_agent_start"])
    write_int(fout, data["has_useful_vel_info"])
    write_int(fout, data["step_rand_int"])

    serialize_randgen(fout, [data["asset_rand_is_seeded"], data["asset_rand_str"]])

    write_int(fout, data["main_width"])
    write_int(fout, data["main_height"])
    write_int(fout, data["out_of_bounds_object"])

    write_float(fout, data["unit"])
    write_float(fout, data["view_dim"])
    write_float(fout, data["x_off"])
    write_float(fout, data["y_off"])
    write_float(fout, data["visibility"])
    write_float(fout, data["min_visibility"])

    write_int(fout, data["grid_w"])
    write_int(fout, data["grid_h"])
    write_vector_int(fout, data["grid_data"])

    if "coinrun" in game_name:
        write_float(fout, data["last_agent_y"])
        write_int(fout, data["wall_theme"])
        write_bool(fout, data["has_support"])
        write_bool(fout, data["facing_right"])
        write_bool(fout, data["is_on_crate"])
        write_float(fout, data["gravity"])
        write_float(fout, data["air_control"])

    elif "maze" in game_name:
        write_int(fout, data["maze_dim"])
        write_int(fout, data["world_dim"])

    import ctypes

    END_OF_BUFFER = ctypes.c_int32(0xCAFECAFE)
    write_int(fout, END_OF_BUFFER.value)

    fout.close()
