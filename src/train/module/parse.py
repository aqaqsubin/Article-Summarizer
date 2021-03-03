def ParseBoolean (arg):
    if len(arg) < 1:
        raise ValueError ('Cannot parse empty string into boolean.')
    arg = arg[0].lower()
    if arg == 'true' or arg == 't':
        return True
    if arg == 'false' or arg == 'f':
        return False
    raise ValueError ('Cannot parse string into boolean.')
