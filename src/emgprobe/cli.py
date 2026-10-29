"""emg-probe COMMAND [options]; `emg-probe COMMAND -h` for the options of one command."""
import importlib
import json
import sys

# command -> (module under emgprobe.protocols, or None for the small commands below; summary)
COMMANDS = {
    'ports': (None, 'List USB serial ports (find each band)'),
    'view': ('view', 'Live plot of raw EMG and IMU of one band'),
    'check': ('check', '30 s rest / fist / rest signal check of one band'),
    'map': ('muscle_map', 'Channel -> muscle map for one or two bands (terminal, ~5 min)'),
    'log': ('contact_log', 'Free-form logging with ENTER-marked contact or phases'),
    'direction': ('direction', 'Stiffness-direction scenarios: mapping, then N trials per scenario'),
    'gesture': ('gesture', 'Rest / fist / open gesture training, held-out test, live prediction (GUI)'),
    'effort': ('effort', 'Subjective effort levels and ramps (GUI)'),
    'timing': (None, 'Per-band clock check of a saved two-band session'),
}


def usage():
    lines = [__doc__, '', 'commands:']
    lines += [f'  {name:<10} {summary}' for name, (_, summary) in COMMANDS.items()]
    return '\n'.join(lines)


def ports():
    from emgprobe.device import usb_ports
    found = usb_ports()
    for p in found:print(f'{p.device:<32} {p.description} | serial {p.serial_number} | location {p.location}')
    if not found:print('No USB serial ports found.')


def timing(argv):
    from emgprobe.bands import session_timing
    if len(argv) != 1:raise SystemExit('usage: emg-probe timing <session folder with bands>')
    print(json.dumps(session_timing(argv[0]), indent=2))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ('-h', '--help'):print(usage());return
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:raise SystemExit(f'unknown command {name!r}\n\n'+usage())
    if name == 'ports':return ports()
    if name == 'timing':return timing(rest)
    return importlib.import_module(f'emgprobe.protocols.{COMMANDS[name][0]}').main(rest)


if __name__ == '__main__':
    main()
