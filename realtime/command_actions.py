def execute_action(command):
    if command == "YES":
        return "✅ Confirmation received."

    elif command == "NO":
        return "❌ Request denied."

    elif command == "HELP":
        return "🚨 EMERGENCY ALERT! Caregiver notified."

    elif command == "WATER":
        return "💧 Water request sent to caregiver."

    else:
        return "⚠️ Unknown command."
