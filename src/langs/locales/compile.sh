#!/bin/sh
# =========================================================
# Скрипт для компиляции .po файлов в .mo
# Работает в Linux, WSL, Docker
# =========================================================

DIR=$(cd "$(dirname "$0")" && pwd)

DATE=$(date +%s)
FILENAME="messages_${DATE}.mo"

LANGUAGES="ru_RU en_US es_ES pt_PT tr_TR id_ID"

printf "\nDeleting old *.mo files...\n"
find "${DIR}" -iname '*.mo' -exec rm -f {} \;

printf "\nCompiling *.mo files...\n"
for LANG in $LANGUAGES; do
  PO_FILE="${DIR}/${LANG}/LC_MESSAGES/messages.po"
  MO_FILE="${DIR}/${LANG}/LC_MESSAGES/${FILENAME}"

  if [ -f "${PO_FILE}" ]; then
    msgfmt -c "${PO_FILE}" -o "${MO_FILE}"
    echo "Compiled ${LANG}"
  else
    echo "Missing ${PO_FILE}, skipped."
  fi
done

echo "${DATE}" > "${DIR}/version.txt"

printf "\nNew version: ${DATE}\n"
printf "Done.\n"
