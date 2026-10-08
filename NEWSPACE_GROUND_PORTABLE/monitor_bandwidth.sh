#!/usr/bin/env bash
set -euo pipefail

JETSON_IP="${JETSON_IP:-192.168.144.16}"
IFACE="$(ip route get "$JETSON_IP" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')"

if [[ -z "${IFACE:-}" || ! -d "/sys/class/net/$IFACE" ]]; then
    echo "Could not determine interface used to reach $JETSON_IP" >&2
    exit 1
fi

echo "GROUND bandwidth monitor"
echo "Jetson    : $JETSON_IP"
echo "Interface : $IFACE"
echo "Ctrl+C to stop"
echo

RX0="$(cat "/sys/class/net/$IFACE/statistics/rx_bytes")"
TX0="$(cat "/sys/class/net/$IFACE/statistics/tx_bytes")"

while sleep 1; do
    RX1="$(cat "/sys/class/net/$IFACE/statistics/rx_bytes")"
    TX1="$(cat "/sys/class/net/$IFACE/statistics/tx_bytes")"

    DRX=$((RX1-RX0))
    DTX=$((TX1-TX0))

    awk -v rx="$DRX" -v tx="$DTX" \
      'BEGIN {
        printf "GROUND  RX=%8.1f kB/s  TX=%8.1f kB/s  TOTAL=%8.1f kB/s   RX=%5.2f Mb/s\n",
        rx/1024, tx/1024, (rx+tx)/1024, rx*8/1000000
      }'

    RX0="$RX1"
    TX0="$TX1"
done
