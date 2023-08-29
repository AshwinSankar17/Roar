MAJOR = 0
MINOR = 0
PATCH = 0
PRE_RELEASE = '0'

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])

__package_name__ = 'roar_toolkit'
__contact_names__ = 'Ashwin Sankar'
__contact_emails__ = 'ashwins1211@gmail.com'
__description__ = 'Roar - a toolkit for Indic Speech AI'
__license__ = 'Apache2'
__keywords__ = 'deep learning, machine learning, gpu, NLP, Roar, pytorch, torch, tts, speech, language'
