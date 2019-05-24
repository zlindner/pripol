# converts acl1010 corpus from .xml to .txt

from glob import glob
from xml.etree import cElementTree as et
import os

for filename in glob('corpus_xml/*.xml'):
    with open(filename, 'r', ) as f_xml:
        policy = et.parse(f_xml).getroot()
        policy_text = ''

        for section in list(policy):
            title = section.find('SUBTITLE').text
            text = section.find('SUBTEXT').text

            if title is not None:
                policy_text = policy_text + title

            if text is not None:
                policy_text = policy_text + text

        policy_text = os.linesep.join([s for s in policy_text.splitlines() if s])

        with open('corpus_text/' + filename[11:-4] + '.txt', 'w') as f_txt:
            f_txt.write(policy_text)  