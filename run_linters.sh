black mourat/

isort mourat/

printf "\nPress any key to continue to pylint...\n"
read -n 1 -s -r
pylint mourat/

printf "\nPress any key to continue to mypy...\n"
read -n 1 -s -r
mypy mourat/
