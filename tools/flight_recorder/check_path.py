import os
import re
import sys
import stat

__all__ = []


PATH_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9/.-]")
MAX_READ_FILE_SIZE_4G = 4294967296  # 4G, 4 * 1024 * 1024 * 1024
MAX_READ_FILE_SIZE_32G = 34359738368  # 32G, 32 * 1024 * 1024 * 1024
MAX_READ_FILE_SIZE_512G = 549755813888  # 512G, 512 * 1024 * 1024 * 1024

# group not writable, others no permission, max stat is 750
WRITE_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH | stat.S_IROTH | stat.S_IXOTH
# group not writable, others not writable, max stat is 755
READ_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH


def type_to_str(value_type):
    return ' or '.join(ii.__name__ for ii in value_type) if isinstance(value_type, tuple) else value_type.__name__


def check_type(value, value_type, param_name="value"):
    if not isinstance(value, value_type):
        raise TypeError('{} must be {}, not {}.'.format(param_name, type_to_str(value_type), type(value).__name__))
    

def get_valid_path(path):
    check_type(path, str, "path")
    if not path or len(path) == 0:
        raise ValueError("The value of the path cannot be empty.")
    if PATH_WHITE_LIST_REGEX.search(path):  # Check special char
        raise ValueError("Input path contains invalid characters.")  # Not printing out the path value for invalid char
    path = os.path.expanduser(path)
    if os.path.islink(os.path.abspath(path)):  # when checking link, get rid of the "/" at the path tail if any
        raise ValueError("The value of the path cannot be a symbolic link: {}.".format(path))

    real_path = os.path.realpath(path)

    if len(real_path) > 4096:
        raise ValueError("The length of file path should be less than 4096.")

    if real_path != path and PATH_WHITE_LIST_REGEX.search(real_path):  # Check special char again
        raise ValueError("Input path contains invalid characters.")  # Not printing out the path value for invalid char

    return real_path


def is_belong_to_user_or_group(file_stat):
    return file_stat.st_uid == os.getuid() or file_stat.st_gid in os.getgroups()


def get_valid_read_path(path, size_max=MAX_READ_FILE_SIZE_4G, check_user_stat=True, is_dir=False):
    real_path = get_valid_path(path)
    if not os.path.isfile(real_path):
        raise ValueError("The path {} doesn't exists or not a file.".format(path))

    file_stat = os.stat(real_path)
    if check_user_stat and not sys.platform.startswith("win") and not is_belong_to_user_or_group(file_stat):
        raise ValueError("The file {} doesn't belong to the current user or group.".format(path))
    if check_user_stat and os.stat(path).st_mode & READ_FILE_NOT_PERMITTED_STAT > 0:
        raise ValueError("The file {} is group writable, or is others writable.".format(path))
    if not os.access(real_path, os.R_OK) or file_stat.st_mode & stat.S_IRUSR == 0:  # At least been 400
        raise ValueError("Current user doesn't have read permission to the file {}.".format(path))
    if not is_dir and size_max > 0 and file_stat.st_size > size_max:
        raise ValueError("The file {} exceeds size limitation of {}.".format(path, size_max))
    return real_path