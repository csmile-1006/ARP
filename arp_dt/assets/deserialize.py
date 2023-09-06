import struct

INT_SIZE = 4
FLOAT_SIZE = 4


def read_int(bytes_string, cursor=0):
    return struct.unpack("i", bytes_string[cursor : cursor + INT_SIZE])[0], cursor + INT_SIZE


def read_bool(bytes_string, cursor=0):
    val, cursor = read_int(bytes_string, cursor=cursor)
    return val > 0, cursor


def read_string(bytes_string, cursor=0):
    string_size, cursor = read_int(bytes_string, cursor=cursor)
    return (
        struct.unpack(f"<{string_size}s", bytes_string[cursor : cursor + string_size])[0].decode(),
        cursor + string_size,
    )


def read_float(bytes_string, cursor=0):
    return struct.unpack("f", bytes_string[cursor : cursor + FLOAT_SIZE])[0], cursor + FLOAT_SIZE


def read_vector_int(bytes_string, cursor=0):
    vector_size, cursor = read_int(bytes_string=bytes_string, cursor=cursor)
    vectors = []
    for i in range(vector_size):
        val, cursor = read_int(bytes_string, cursor=cursor)
        vectors.append(val)
    return vectors, cursor


def deserialize_randgen(bytes_string, cursor=0):
    is_seeded, cursor = read_int(bytes_string, cursor)
    _str, cursor = read_string(bytes_string, cursor)
    return is_seeded, _str, cursor


def read_entities(bytes_string, cursor=0):
    ents = []
    num_entities, cursor = read_int(bytes_string=bytes_string, cursor=cursor)
    for i in range(num_entities):
        entity_info, cursor = deserialize_entity(bytes_string, cursor)
        ents.append(entity_info)

    return ents, cursor


def deserialize_entity(bytes_string, cursor=0):
    x, cursor = read_float(bytes_string, cursor)
    y, cursor = read_float(bytes_string, cursor)

    vx, cursor = read_float(bytes_string, cursor)
    vy, cursor = read_float(bytes_string, cursor)

    rx, cursor = read_float(bytes_string, cursor)
    ry, cursor = read_float(bytes_string, cursor)

    type, cursor = read_int(bytes_string, cursor)
    image_type, cursor = read_int(bytes_string, cursor)
    image_theme, cursor = read_int(bytes_string, cursor)

    render_z, cursor = read_int(bytes_string, cursor)

    will_erase, cursor = read_int(bytes_string, cursor)
    collides_with_entities, cursor = read_int(bytes_string, cursor)

    collision_margin, cursor = read_float(bytes_string, cursor)
    rotation, cursor = read_float(bytes_string, cursor)
    vrot, cursor = read_float(bytes_string, cursor)

    is_reflected, cursor = read_int(bytes_string, cursor)
    fire_time, cursor = read_int(bytes_string, cursor)
    spawn_time, cursor = read_int(bytes_string, cursor)
    life_time, cursor = read_int(bytes_string, cursor)
    expire_time, cursor = read_int(bytes_string, cursor)
    use_abs_coords, cursor = read_int(bytes_string, cursor)

    friction, cursor = read_float(bytes_string, cursor)
    smart_step, cursor = read_int(bytes_string, cursor)
    avoids_collisions, cursor = read_int(bytes_string, cursor)
    auto_erase, cursor = read_int(bytes_string, cursor)

    alpha, cursor = read_float(bytes_string, cursor)
    health, cursor = read_float(bytes_string, cursor)
    theta, cursor = read_float(bytes_string, cursor)
    grow_rate, cursor = read_float(bytes_string, cursor)
    alpha_decay, cursor = read_float(bytes_string, cursor)
    climber_spawn_x, cursor = read_float(bytes_string, cursor)

    entity_info = {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "rx": rx,
        "ry": ry,
        "type": type,
        "image_type": image_type,
        "image_theme": image_theme,
        "render_z": render_z,
        "will_erase": will_erase,
        "collides_with_entities": collides_with_entities,
        "collision_margin": collision_margin,
        "rotation": rotation,
        "vrot": vrot,
        "is_reflected": is_reflected,
        "fire_time": fire_time,
        "spawn_time": spawn_time,
        "life_time": life_time,
        "expire_time": expire_time,
        "use_abs_coords": use_abs_coords,
        "friction": friction,
        "smart_step": smart_step,
        "avoids_collisions": avoids_collisions,
        "auto_erase": auto_erase,
        "alpha": alpha,
        "health": health,
        "theta": theta,
        "grow_rate": grow_rate,
        "alpha_decay": alpha_decay,
        "climber_spawn_x": climber_spawn_x,
    }

    return entity_info, cursor


def deserialize(bytes_string, env_type="none"):
    cursor = 0

    SERIALIZE_VERSION, cursor = read_int(bytes_string, cursor=cursor)
    game_name, cursor = read_string(bytes_string, cursor=cursor)
    paint_vel_info, cursor = read_int(bytes_string, cursor=cursor)
    use_generated_assets, cursor = read_int(bytes_string, cursor=cursor)
    use_monochrome_assets, cursor = read_int(bytes_string, cursor=cursor)
    restrict_themes, cursor = read_int(bytes_string, cursor=cursor)
    use_backgrounds, cursor = read_int(bytes_string, cursor=cursor)
    center_agent, cursor = read_int(bytes_string, cursor=cursor)
    debug_mode, cursor = read_int(bytes_string, cursor=cursor)
    distribution_mode, cursor = read_int(bytes_string, cursor=cursor)
    use_sequential_levels, cursor = read_int(bytes_string, cursor=cursor)

    if "_" in game_name or env_type == "aisc":
        random_percent, cursor = read_int(bytes_string, cursor=cursor)
        key_penalty, cursor = read_int(bytes_string, cursor=cursor)
        step_penalty, cursor = read_int(bytes_string, cursor=cursor)
        rand_region, cursor = read_int(bytes_string, cursor=cursor)
        continue_after_coin, cursor = read_int(bytes_string, cursor=cursor)

    use_easy_jump, cursor = read_int(bytes_string, cursor=cursor)
    plain_assets, cursor = read_int(bytes_string, cursor=cursor)
    physics_mode, cursor = read_int(bytes_string, cursor=cursor)

    grid_step, cursor = read_int(bytes_string, cursor=cursor)
    level_seed_low, cursor = read_int(bytes_string, cursor=cursor)
    level_seed_high, cursor = read_int(bytes_string, cursor=cursor)
    game_type, cursor = read_int(bytes_string, cursor=cursor)
    game_n, cursor = read_int(bytes_string, cursor=cursor)

    # RandGen deserialize
    level_seed_is_seeded, level_seed_str, cursor = deserialize_randgen(bytes_string, cursor=cursor)
    rand_is_seeded, rand_str, cursor = deserialize_randgen(bytes_string, cursor=cursor)

    step_data_reward, cursor = read_float(bytes_string, cursor=cursor)
    step_data_done, cursor = read_int(bytes_string, cursor=cursor)
    step_data_level_complete, cursor = read_int(bytes_string, cursor=cursor)

    action, cursor = read_int(bytes_string, cursor=cursor)
    timeout, cursor = read_int(bytes_string, cursor=cursor)

    current_level_seed, cursor = read_int(bytes_string, cursor=cursor)
    prev_level_seed, cursor = read_int(bytes_string, cursor=cursor)
    episodes_remaining, cursor = read_int(bytes_string, cursor=cursor)
    episodes_done, cursor = read_int(bytes_string, cursor=cursor)

    last_reward_timer, cursor = read_int(bytes_string, cursor=cursor)
    last_reward, cursor = read_float(bytes_string, cursor=cursor)
    default_action, cursor = read_int(bytes_string, cursor=cursor)

    fixed_asset_seed, cursor = read_int(bytes_string, cursor=cursor)

    cur_time, cursor = read_int(bytes_string, cursor=cursor)
    is_waiting_for_sleep, cursor = read_int(bytes_string, cursor=cursor)

    grid_size, cursor = read_int(bytes_string, cursor=cursor)
    entities, cursor = read_entities(bytes_string, cursor=cursor)

    use_procgen_background, cursor = read_int(bytes_string, cursor=cursor)
    background_index, cursor = read_int(bytes_string, cursor=cursor)
    bg_tile_ratio, cursor = read_float(bytes_string, cursor=cursor)
    bg_pct_x, cursor = read_float(bytes_string, cursor=cursor)

    char_dim, cursor = read_float(bytes_string, cursor=cursor)
    last_move_action, cursor = read_int(bytes_string, cursor=cursor)
    move_action, cursor = read_int(bytes_string, cursor=cursor)
    special_action, cursor = read_int(bytes_string, cursor=cursor)
    mixrate, cursor = read_float(bytes_string, cursor=cursor)
    maxspeed, cursor = read_float(bytes_string, cursor=cursor)
    max_jump, cursor = read_float(bytes_string, cursor=cursor)

    action_vx, cursor = read_float(bytes_string, cursor=cursor)
    action_vy, cursor = read_float(bytes_string, cursor=cursor)
    action_vrot, cursor = read_float(bytes_string, cursor=cursor)

    center_x, cursor = read_float(bytes_string, cursor=cursor)
    center_y, cursor = read_float(bytes_string, cursor=cursor)

    random_agent_start, cursor = read_int(bytes_string, cursor=cursor)
    has_useful_vel_info, cursor = read_int(bytes_string, cursor=cursor)
    step_rand_int, cursor = read_int(bytes_string, cursor=cursor)

    asset_rand_is_seeded, asset_rand_str, cursor = deserialize_randgen(bytes_string, cursor=cursor)

    main_width, cursor = read_int(bytes_string, cursor=cursor)
    main_height, cursor = read_int(bytes_string, cursor=cursor)
    out_of_bounds_object, cursor = read_int(bytes_string, cursor=cursor)

    unit, cursor = read_float(bytes_string, cursor=cursor)
    view_dim, cursor = read_float(bytes_string, cursor=cursor)
    x_off, cursor = read_float(bytes_string, cursor=cursor)
    y_off, cursor = read_float(bytes_string, cursor=cursor)
    visibility, cursor = read_float(bytes_string, cursor=cursor)
    min_visibility, cursor = read_float(bytes_string, cursor=cursor)

    grid_w, cursor = read_int(bytes_string, cursor=cursor)
    grid_h, cursor = read_int(bytes_string, cursor=cursor)
    grid_data, cursor = read_vector_int(bytes_string, cursor=cursor)

    data = {
        "SERIALIZE_VERSION": SERIALIZE_VERSION,
        "game_name": game_name,
        "paint_vel_info": paint_vel_info,
        "use_generated_assets": use_generated_assets,
        "use_monochrome_assets": use_monochrome_assets,
        "restrict_themes": restrict_themes,
        "use_backgrounds": use_backgrounds,
        "center_agent": center_agent,
        "debug_mode": debug_mode,
        "distribution_mode": distribution_mode,
        "use_sequential_levels": use_sequential_levels,
        "use_easy_jump": use_easy_jump,
        "plain_assets": plain_assets,
        "physics_mode": physics_mode,
        "grid_step": grid_step,
        "level_seed_low": level_seed_low,
        "level_seed_high": level_seed_high,
        "game_type": game_type,
        "game_n": game_n,
        "level_seed_is_seeded": level_seed_is_seeded,
        "level_seed_str": level_seed_str,
        "rand_is_seeded": rand_is_seeded,
        "rand_str": rand_str,
        "step_data_reward": step_data_reward,
        "step_data_done": step_data_done,
        "step_data_level_complete": step_data_level_complete,
        "action": action,
        "timeout": timeout,
        "current_level_seed": current_level_seed,
        "prev_level_seed": prev_level_seed,
        "episodes_remaining": episodes_remaining,
        "episodes_done": episodes_done,
        "last_reward_timer": last_reward_timer,
        "last_reward": last_reward,
        "default_action": default_action,
        "fixed_asset_seed": fixed_asset_seed,
        "cur_time": cur_time,
        "is_waiting_for_sleep": is_waiting_for_sleep,
        "grid_size": grid_size,
        "entities": entities,
        "use_procgen_background": use_procgen_background,
        "background_index": background_index,
        "bg_tile_ratio": bg_tile_ratio,
        "bg_pct_x": bg_pct_x,
        "char_dim": char_dim,
        "last_move_action": last_move_action,
        "move_action": move_action,
        "special_action": special_action,
        "mixrate": mixrate,
        "maxspeed": maxspeed,
        "max_jump": max_jump,
        "action_vx": action_vx,
        "action_vy": action_vy,
        "action_vrot": action_vrot,
        "center_x": center_x,
        "center_y": center_y,
        "random_agent_start": random_agent_start,
        "has_useful_vel_info": has_useful_vel_info,
        "step_rand_int": step_rand_int,
        "asset_rand_is_seeded": asset_rand_is_seeded,
        "asset_rand_str": asset_rand_str,
        "main_width": main_width,
        "main_height": main_height,
        "out_of_bounds_object": out_of_bounds_object,
        "unit": unit,
        "view_dim": view_dim,
        "x_off": x_off,
        "y_off": y_off,
        "visibility": visibility,
        "min_visibility": min_visibility,
        "grid_w": grid_w,
        "grid_h": grid_h,
        "grid_data": grid_data,
    }

    if "_" in game_name or env_type == "aisc":
        aisc_dict = {
            "random_percent": random_percent,
            "key_penalty": key_penalty,
            "step_penalty": step_penalty,
            "rand_region": rand_region,
            "continue_after_coin": continue_after_coin,
        }
        data.update(aisc_dict)

    # coinrun specific
    if "coinrun" in game_name:
        last_agent_y, cursor = read_float(bytes_string, cursor=cursor)
        wall_theme, cursor = read_int(bytes_string, cursor=cursor)
        has_support, cursor = read_bool(bytes_string, cursor=cursor)
        facing_right, cursor = read_bool(bytes_string, cursor=cursor)
        is_on_crate, cursor = read_bool(bytes_string, cursor=cursor)
        gravity, cursor = read_float(bytes_string, cursor=cursor)
        air_control, cursor = read_float(bytes_string, cursor=cursor)
        game_data = {
            "last_agent_y": last_agent_y,
            "wall_theme": wall_theme,
            "has_support": has_support,
            "facing_right": facing_right,
            "is_on_crate": is_on_crate,
            "gravity": gravity,
            "air_control": air_control,
        }

    elif "maze" in game_name:
        maze_dim, cursor = read_int(bytes_string, cursor=cursor)
        world_dim, cursor = read_int(bytes_string, cursor=cursor)
        game_data = {"maze_dim": maze_dim, "world_dim": world_dim}

    data.update(game_data)
    return data
