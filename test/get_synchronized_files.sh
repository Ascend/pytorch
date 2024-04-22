#!/usr/bin/env bash
# shellcheck disable=SC2164
path=$(cd "$(dirname "$0")";pwd)
mkdir temp_tests
# shellcheck disable=SC2164
cd temp_tests
git clone -b release/2.3 https://github.com/pytorch/pytorch.git --depth=1
cd ../..
if [ -e "${path}"/testfiles_synchronized.txt ]; then
     # shellcheck disable=SC2002
     cat "${path}"/testfiles_synchronized.txt | while IFS= read -r line; do
         if [ "${line}" != "" ]; then
             cp -rf "${path}"/temp_tests/pytorch/"${line}" "${line}"
         fi
     done
fi
if [ -e "${path}"/testfolder_synchronized.txt ]; then
     # shellcheck disable=SC2002
     cat "${path}"/testfolder_synchronized.txt | while IFS= read -r line; do
          if [ "${line}" != "" ]; then
               cp -rf "${path}"/temp_tests/pytorch/"${line}" "${line%/*}"
          fi
     done
fi
cd ./test
rm -rf temp_tests
