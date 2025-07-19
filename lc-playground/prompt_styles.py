def apply_style(style: str, user_input: str) -> str:
    if style == "direct":
        return f"Answer this clearly: {user_input}"
    elif style == "friendly":
        return f"Hi there! Can you please help me with this? {user_input}"
    elif style == "elaborate":
        return f"Provide a detailed and thoughtful response to: {user_input}"
    elif style == "expert":
        return f"You are an expert. Answer this professionally: {user_input}"
    else:
        return user_input
