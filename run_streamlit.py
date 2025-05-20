import os
import sys
import subprocess

def main():
    """Launch the Streamlit app."""
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Try to run Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError:
        print("Error running Streamlit. Make sure it's installed.")
        print("You can install it with: pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\nStopping the Streamlit app...")

if __name__ == "__main__":
    main() 