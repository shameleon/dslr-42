#!/bin/bash

activate () {
    . $PWD/venv/bin/activate
}

help()
{
    echo "Usage: install venv  [ -i | --install ]
                 activate venv [ -a | --activate ]
                 clean pycache [ -c | --clean ]
                 help          [ -h | --help  ]"
    exit 2
}

# script name
echo $0
PWD=`pwd`
# https://stackabuse.com/how-to-parse-command-line-arguments-in-bash/
# $1 first arg

SHORT=i:,a:,c:,
LONG=install:,activate:,clean:,help
OPTS=$(getopt --alternative --name weather --options $SHORT --longoptions $LONG -- "$@")

VALID_ARGUMENTS=$# # Returns the count of arguments that are in short or long options

if [ "$VALID_ARGUMENTS" -eq 0 ]; then
  help
fi

eval set -- "$OPTS"

while :
do
  case "$1" in
    -i | --install )
      echo "install"
      shift
      ;;
    -a | --activate )
      echo "activate"
      shift
      ;;
    -c | --clean )
      echo "clean"
      shift
      ;;
    -h | --help )
      help
      exit 2
      ;;
    --)
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1"
      ;;
  esac
done

