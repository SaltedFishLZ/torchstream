"""
Helper functions related to file systems
"""
import os
import platform

def strip_extension(path_to_file):
    """
    Helps you remove the file extension from its' path
    """
    return os.path.splitext(path_to_file)[0]

def strip_dirpath(path_to_file):
    """
    Helps you remove the directory path from a file's path
    """
    return os.path.basename(path_to_file)

def modification_date(path_to_file):
    """
    Give you the latest modification date of a file
    """
    return os.path.getmtime(path_to_file)

def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    stat = os.stat(path_to_file)
    try:
        return stat.st_birthtime
    except AttributeError:
        # We're probably on Linux. No easy way to get creation dates here,
        # so we'll settle for when its content was last modified.
        return stat.st_mtime

def touch_date(path_to_file):
    """
    Conservative touch date: the latest data of create/modification date
    """
    return max(creation_date(path_to_file), modification_date(path_to_file))
