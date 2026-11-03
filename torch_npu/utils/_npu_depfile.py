import os
import sys


def _parse_dependencies(content):
    content = content.replace("\\\r\n", "").replace("\\\n", "")
    escaped = False
    for index, character in enumerate(content):
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == ":":
            break
    else:
        raise ValueError("depfile does not contain a target")

    dependencies = []
    token = []
    index += 1
    while index < len(content):
        character = content[index]
        if character == "\\" and index + 1 < len(content):
            index += 1
            token.append(content[index])
        elif character == "$" and index + 1 < len(content) and content[index + 1] == "$":
            token.append("$")
            index += 1
        elif character.isspace():
            if token:
                dependencies.append("".join(token))
                token = []
        else:
            token.append(character)
        index += 1
    if token:
        dependencies.append("".join(token))
    return dependencies


def _escape_path(path):
    return (
        path.replace("\\", "\\\\")
        .replace("$", "$$")
        .replace(" ", "\\ ")
        .replace("\t", "\\\t")
        .replace("#", "\\#")
        .replace(":", "\\:")
    )


def _dependency_path(path):
    if not os.path.exists(path):
        return None
    if not os.path.exists(os.path.normpath(path)):
        return os.path.realpath(path)
    return path


def filter_depfile(source, destination, target):
    with open(source, encoding="utf-8") as input_file:
        dependencies = _parse_dependencies(input_file.read())
    dependencies = [_dependency_path(path) for path in dependencies]
    dependencies = [path for path in dependencies if path is not None]

    temporary = destination + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output_file:
        output_file.write(f"{_escape_path(target)}:")
        for dependency in dependencies:
            output_file.write(f" {_escape_path(dependency)}")
        output_file.write("\n")
    os.replace(temporary, destination)
    os.remove(source)


if __name__ == "__main__":
    filter_depfile(*sys.argv[1:])
