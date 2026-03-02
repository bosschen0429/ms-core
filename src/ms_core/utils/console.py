import builtins
import sys


def safe_print(*args, **kwargs):
    """
    Print with encoding fallback to avoid UnicodeEncodeError on Windows consoles.
    """
    file = kwargs.get("file", sys.stdout)
    encoding = getattr(file, "encoding", None) or "utf-8"
    safe_args = []
    for arg in args:
        text = str(arg)
        try:
            text = text.encode(encoding, errors="replace").decode(encoding)
        except Exception:
            text = text.encode("utf-8", errors="replace").decode("utf-8")
        safe_args.append(text)
    builtins.print(*safe_args, **kwargs)
