import gym3
from procgen_aisc import ProcgenGym3Env as ProcgenAISCGym3Env
from procgen_highres_aisc import ProcgenGym3Env as ProcgenHighResAISCGym3Env


def get_procgen_venv(*, env_id, num_envs, rendering=False, high_res=False, env_type="none", **env_kwargs):
    if rendering:
        env_kwargs["render_human"] = True

    if high_res:
        if env_type == "none":
            # env = ProcgenHighResGym3Env(num=num_envs, env_name=env_id, **env_kwargs)
            env = ProcgenHighResAISCGym3Env(num=num_envs, env_name=env_id, **env_kwargs)
        else:
            env = ProcgenHighResAISCGym3Env(num=num_envs, env_name=f"{env_id}_{env_type}", **env_kwargs)
    else:
        if env_type == "none":
            # env = ProcgenGym3Env(num=num_envs, env_name=env_id, **env_kwargs)
            env = ProcgenAISCGym3Env(num=num_envs, env_name=env_id, **env_kwargs)
        else:
            env = ProcgenAISCGym3Env(num=num_envs, env_name=f"{env_id}_{env_type}", **env_kwargs)

    env = gym3.ExtractDictObWrapper(env, "rgb")

    if rendering:
        env = gym3.ViewerWrapper(env, info_key="rgb")
    return env


def get_venv(num_envs, env_name, **env_kwargs):
    venv = get_procgen_venv(num_envs=num_envs, env_id=env_name, **env_kwargs)

    return venv
