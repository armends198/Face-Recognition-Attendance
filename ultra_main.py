"""
ULTRA-ACCURATE FACE ATTENDANCE SYSTEM
Main Menu - Central Control Panel
"""

import os
import sys
import subprocess

def clear_screen():
    """Clear console screen"""
    try:
        # Set TERM if not set (prevents warning on some systems)
        if 'TERM' not in os.environ:
            os.environ['TERM'] = 'xterm'

        os.system('cls' if os.name == 'nt' else 'clear')
    except:
        # Fallback: just print newlines if clear fails
        print('\n' * 50)

def print_header():
    """Print system header"""
    print("="*70)
    print("   ULTRA-ACCURATE FACE ATTENDANCE SYSTEM")
    print("   Deep Learning Recognition (VGG-Face)")
    print("="*70)

def print_menu():
    """Print main menu"""
    print("\n📋 MAIN MENU:")
    print()
    print("  [1] 🧪 Test Installation")
    print("  [2] 👤 Register New Person")
    print("  [3] 🎓 Train System")
    print("  [4] 📸 Start Attendance")
    print("  [5] 📊 View Attendance Records")
    print("  [6] 👥 List Registered People")
    print("  [7] 🔄 Reset System (Delete All Data)")
    print("  [0] ❌ Exit")
    print()
    print("="*70)

def run_script(script_name):
    """Run a Python script"""
    try:
        if os.path.exists(script_name):
            print(f"\n🚀 Running {script_name}...\n")
            subprocess.run([sys.executable, script_name])
        else:
            print(f"\n❌ Error: {script_name} not found!")
            print(f"   Please make sure {script_name} is in the same folder.")
        input("\n\nPress Enter to continue...")
    except Exception as e:
        print(f"\n❌ Error running {script_name}: {e}")
        input("\n\nPress Enter to continue...")

def view_attendance():
    """View attendance records"""
    clear_screen()
    print_header()
    print("\n📊 ATTENDANCE RECORDS\n")

    if not os.path.exists('ultra_attendance.csv'):
        print("❌ No attendance records found.")
        print("   Run 'Start Attendance' first to create records.")
    else:
        try:
            with open('ultra_attendance.csv', 'r') as f:
                lines = f.readlines()

            if len(lines) <= 1:
                print("📝 Attendance file exists but is empty.")
            else:
                print(f"Total entries: {len(lines) - 1}\n")
                print("-"*70)

                # Print header
                print(lines[0].strip())
                print("-"*70)

                # Print last 20 entries
                display_lines = lines[-20:] if len(lines) > 20 else lines[1:]

                if len(lines) > 21:
                    print(f"... (showing last 20 of {len(lines)-1} entries) ...")
                    print()

                for line in display_lines:
                    if line.strip():
                        print(line.strip())

                print("-"*70)

        except Exception as e:
            print(f"❌ Error reading attendance file: {e}")

    input("\n\nPress Enter to continue...")

def list_registered_people():
    """List all registered people"""
    clear_screen()
    print_header()
    print("\n👥 REGISTERED PEOPLE\n")

    if not os.path.exists('face_database'):
        print("❌ No database folder found.")
        print("   Register people first using option [2].")
    else:
        person_folders = [f for f in os.listdir('face_database')
                         if os.path.isdir(os.path.join('face_database', f))]

        if not person_folders:
            print("📝 Database folder exists but no people registered yet.")
            print("   Use option [2] to register people.")
        else:
            print(f"Total registered: {len(person_folders)}\n")
            print("-"*70)

            for i, folder in enumerate(person_folders, 1):
                # Parse folder name (format: ID_Name)
                parts = folder.split('_', 1)
                if len(parts) == 2:
                    person_id, person_name = parts

                    # Check if embeddings exist
                    embedding_file = os.path.join('face_database', folder, 'embeddings.pkl')
                    status = "✓ Trained" if os.path.exists(embedding_file) else "⚠ Not trained"

                    # Count samples
                    samples = len([f for f in os.listdir(os.path.join('face_database', folder))
                                 if f.endswith('.jpg')])

                    print(f"  {i}. {person_name}")
                    print(f"     ID: {person_id} | Samples: {samples} | Status: {status}")
                    print()
                else:
                    print(f"  {i}. {folder} (Unknown format)")
                    print()

            print("-"*70)

            # Check if trained database exists
            if os.path.exists('ultra_database.pkl'):
                print("\n✓ Trained database file (ultra_database.pkl) exists")
            else:
                print("\n⚠ Trained database not found - Run option [3] to train")

    input("\n\nPress Enter to continue...")

def confirm_reset():
    """Reset system with confirmation"""
    clear_screen()
    print_header()
    print("\n🔄 RESET SYSTEM\n")
    print("⚠️  WARNING: This will DELETE ALL DATA!")
    print()
    print("This includes:")
    print("  - All registered people (face_database/)")
    print("  - Trained model (ultra_database.pkl)")
    print("  - Attendance records (ultra_attendance.csv)")
    print()
    print("-"*70)

    response = input("\nType 'YES' to confirm reset: ").strip()

    if response == "YES":
        run_script('ultra_reset.py')
    else:
        print("\n❌ Reset cancelled.")
        input("\nPress Enter to continue...")

def check_system_status():
    """Check and display system status"""
    status = {
        'test_script': os.path.exists('test_installation.py'),
        'register_script': os.path.exists('ultra_register.py'),
        'train_script': os.path.exists('ultra_train.py'),
        'attendance_script': os.path.exists('ultra_attendance.py'),
        'reset_script': os.path.exists('ultra_reset.py'),
        'database_folder': os.path.exists('face_database'),
        'trained_model': os.path.exists('ultra_database.pkl'),
        'attendance_file': os.path.exists('ultra_attendance.csv')
    }

    print("\n📊 System Status:")
    print()

    # Check scripts
    scripts_ok = all([status['test_script'], status['register_script'],
                     status['train_script'], status['attendance_script'],
                     status['reset_script']])

    if scripts_ok:
        print("  ✓ All required scripts present")
    else:
        print("  ⚠ Missing scripts:")
        if not status['test_script']: print("    - test_installation.py")
        if not status['register_script']: print("    - ultra_register.py")
        if not status['train_script']: print("    - ultra_train.py")
        if not status['attendance_script']: print("    - ultra_attendance.py")
        if not status['reset_script']: print("    - ultra_reset.py")

    # Check database
    if status['database_folder']:
        person_count = len([f for f in os.listdir('face_database')
                           if os.path.isdir(os.path.join('face_database', f))])
        print(f"  ✓ Database folder exists ({person_count} people registered)")
    else:
        print("  • Database folder not created yet")

    # Check trained model
    if status['trained_model']:
        print("  ✓ Trained model exists")
    else:
        print("  • No trained model (run option [3] after registration)")

    # Check attendance
    if status['attendance_file']:
        try:
            with open('ultra_attendance.csv', 'r') as f:
                entry_count = len(f.readlines()) - 1
            print(f"  ✓ Attendance file exists ({entry_count} entries)")
        except:
            print("  ✓ Attendance file exists")
    else:
        print("  • No attendance records yet")

    print()

def main():
    """Main program loop"""
    while True:
        clear_screen()
        print_header()
        check_system_status()
        print_menu()

        choice = input("Enter your choice [0-7]: ").strip()

        if choice == '1':
            run_script('test_installation.py')

        elif choice == '2':
            run_script('ultra_register.py')

        elif choice == '3':
            run_script('ultra_train.py')

        elif choice == '4':
            run_script('ultra_attendance.py')

        elif choice == '5':
            view_attendance()

        elif choice == '6':
            list_registered_people()

        elif choice == '7':
            confirm_reset()

        elif choice == '0':
            clear_screen()
            print("\n👋 Thank you for using Ultra-Accurate Face Attendance System!")
            print("   Goodbye!\n")
            sys.exit(0)

        else:
            print("\n❌ Invalid choice. Please enter a number from 0 to 7.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clear_screen()
        print("\n\n👋 Program interrupted. Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)
