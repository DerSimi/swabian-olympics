#!/bin/bash

# Call this script like this:
#   ./competition_autorestart.sh --args "--agent=<your-agent-name>"

# Wrapper script for the competition code that automatically restarts the client if it terminates.
# It keeps track of the number of restarts and aborts if there are too many
# within a certain time window (see variables below).
#
# NOTE: If you want to stop the script, you need to press Ctrl+C twice.
# Once to stop the client and then again to stop the wrapper script.

# load .env file
if [ -f ".env" ]; then
    source .env
else
    echo "No .env file was found in the working directory (did you copy and complete from .env.sample?)"
fi

# Keep track of restarts within this time window
THRESHOLD_TIME_WINDOW=600  # 10 min
# Threshold for the number of restarts in the time window
THRESHOLD=10

# If you set a topic name here, restart notifications will be sent to
# ntfy.sh/$NFTY_TOPIC (so you can more easily monitor if there are problems).
NTFY_TOPIC=$NTFY_TOPIC

# Array to hold timestamps of terminations
termination_times=()

if [ -n "${NTFY_TOPIC}" ]; then
    curl -H "Priority: urgent" -H "Tags: warning" \
        -d "Initializing autostart competition mode ${THRESHOLD_TIME_WINDOW}" \
        ntfy.sh/${NTFY_TOPIC}
fi

while true; do

    # Run the command foobar
    python3 src/run_competition.py "$@"

    # Get the current timestamp
    current_time=$(date +%s)

    # Add the current timestamp to the termination times array
    termination_times+=("$current_time")

    # Remove timestamps outside of the time window
    termination_times=($(for time in "${termination_times[@]}"; do
        if (( current_time - time <= ${THRESHOLD_TIME_WINDOW} )); then
            echo "$time"
        fi
    done))

    # Check if the number of terminations exceeds the threshold
    if (( ${#termination_times[@]} > ${THRESHOLD} )); then
        echo
        echo "##############################################"
        echo "Restarted too many times within the last ${THRESHOLD_TIME_WINDOW} seconds."

        if [ -n "${NTFY_TOPIC}" ]; then
            curl -H "Priority: urgent" -H "Tags: warning" \
                -d "Too many restarts within the last ${THRESHOLD_TIME_WINDOW}" \
                ntfy.sh/${NTFY_TOPIC}
        fi

        exit 1
    fi

    # Wait a bit before restarting.  In case the client terminated due to a
    # server restart, it will take some time before the server is ready again.
    sleep 20

    echo
    echo "##############################################"
    echo "#                Restarting                  #"
    echo "##############################################"
    echo

    if [ -n "${NTFY_TOPIC}" ]; then
        curl -d "Restarting" ntfy.sh/${NTFY_TOPIC}
    fi
done

