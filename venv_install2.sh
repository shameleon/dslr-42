#!/bin/bash

help()
{
    echo "Usage: install venv  [ -i | --install ]
    activate venv [ -a | --activate ]
    clean pycache [ -c | --clean ]
    help          [ -h | --help  ]"
}

# script name
echo $0 

# https://stackabuse.com/how-to-parse-command-line-arguments-in-bash/
# https://sookocheff.com/post/bash/parsing-bash-script-arguments-with-shopts/
# $1 first arg

#if [ "$OS" = "Darwin" ]
#then
#   echo "MacOSX :)"
# fi

while getopts ":iach" flag;
do
  case "${flag}" in
      i)
        echo "install"
        target=$OPTARG
        echo $OPTARG
        ;;
      a)
        echo "activate"
        ;;
      c)
        echo "clean"
        ;;
      h)
        help
        ;;
      \? )
        echo "Usage: cmd [-i] [-a] [-c]"
        ;;
  esac
done
echo "${target}"
#shift $((OPTIND -1))




