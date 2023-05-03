import curses
import time

# Initialize the screen
screen = curses.initscr()

# Create a window for the header
header = curses.newwin(3, curses.COLS, 0, 0)

# Create a window for the body
body = curses.newwin(curses.LINES-3, curses.COLS, 3, 0)

# Print the header text
header.addstr(0, 0, "Header line 1")
header.addstr(1, 0, "Header line 2")
header.addstr(2, 0, "Header line 3")

# Print the body text
body.addstr(0, 0, "Body line 1")
body.addstr(1, 0, "Body line 2")
body.addstr(2, 0, "Body line 3")

# Refresh the windows to display the text
header.refresh()
body.refresh()

# Wait for 5 seconds
time.sleep(5)

# Clean up and exit
curses.endwin()
