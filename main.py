import tkinter as tk

from ui import OutlineApp


def main() -> None:
    root = tk.Tk()
    OutlineApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
