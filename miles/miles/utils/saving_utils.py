from pathlib import Path


def get_args_save_dir(save_path: str) -> Path:
    """从 args.save（checkpoint 保存路径）推导出 train_args 的保存目录。

    逻辑：取 save_path 的父目录，然后拼接 "train_args"。
    例如：
        save_path = ".../outputs/PROJECT/EXP/ckpt"
        返回      = ".../outputs/PROJECT/EXP/train_args"

    Args:
        save_path: args.save 的值，即 checkpoint 保存路径。

    Returns:
        train_args 的目标保存目录 Path 对象。
    """
    return Path(save_path).parent / "train_args"
