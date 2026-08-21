black project_name/

isort project_name/

printf "\nPress any key to continue to pylint...\n"
read -n 1 -s -r
pylint project_name/

printf "\nPress any key to continue to mypy...\n"
read -n 1 -s -r
mypy project_name/
