# Olivier Georgeon, 2022.
# This code is used to teach Developmental AI.
# Requires:
#   - A PetiCat robot https://github.com/OlivierGeorgeon/osoyoo/wiki

from PetitCatTester import PetitCatTester
import sys
import json


class PetitCatEnacter:
    def __init__(self, robot_ip, timeout=5):
        # Handling the wifi connection to the robot
        self.petitcat_tester = PetitCatTester(robot_ip, timeout)

    def outcome(self, action):
        """ Enacting an action and returning the outcome """
        outcome = 0
        if action in [0, 1, 2]:
            # Convert the action to command and send it to the robot
            command = ['8', '1', '3'][action]
            print("sending:", command)

            outcome_string = self.petitcat_tester.send(command)
            if outcome_string is not None:
                json_outcome = json.loads(outcome_string)
                # Return the outcome based on floor change
                if 'floor' in json_outcome and json_outcome['floor'] > 0:
                    outcome = 1
                if 'impact' in json_outcome and json_outcome['impact'] > 0:
                    outcome = 1
        else:
            print("Action", action, "is not accepted")
            exit()

        return outcome


# Testing the Osoyoo Car Enacter by controlling the robot from the console
# py PetitCatEnacter.py 192.168.8.242
if __name__ == "__main__":
    ip = "192.168.4.1"
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    else:
        print("Please provide your robot's IP address")
    print("Connecting to robot: " + ip)
    print("Enter action: 0: Forward, 1:left, 2:right, anything else abort.")
    e = PetitCatEnacter(ip)

    _outcome = 0
    for i in range(10):
        _action = input("Enter action: ")
        _outcome = e.outcome(int(_action))
        print("Action: " + _action + " Outcome: " + str(_outcome))
