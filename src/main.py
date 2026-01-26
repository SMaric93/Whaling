"""
A simple greeting application.
"""


def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"


def main():
    """Main entry point."""
    user_name = input("Enter your name: ")
    print(greet(user_name))


if __name__ == "__main__":
    main()
