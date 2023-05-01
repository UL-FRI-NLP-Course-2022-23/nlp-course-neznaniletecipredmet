def remove_new_lines(txt: str) -> str:
    # Replace new lines
    txt = txt.replace("\n", " ")

    txt = " ".join(txt.split())

    return txt