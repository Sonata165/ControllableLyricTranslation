'''
Split songs into paragraphs
Generate constraints for input
'''

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../../../'))

from utils_common.utils import ls, jpath, SyllableCounter, BoundaryUtilEn
from tqdm import tqdm

def _main():
    _procedures()


def _procedures():
    split_paragraphs()
    obtain_constraints()

def split_paragraphs():
    '''
    Generate sub-datasets, each set contain a paragraph of original lyrics
    '''
    data_dir = './data'
    out_dir = '../../datasets/real'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    songs = ls(data_dir)
    for song in songs:
        song_path = jpath(data_dir, song)
        with open(song_path, 'r') as f:
            lyrics = f.readlines()
        lyrics = [l.strip() for l in lyrics]
        pars = []
        par = []
        for line in lyrics:
            if len(line) > 0:
                par.append(line)
            else:
                pars.append(par)
                par = []
        if len(par) > 0:
            pars.append(par)

        # Save data
        out_song_dir = jpath(out_dir, song.split('.')[0])
        if not os.path.exists(out_song_dir):
            os.mkdir(out_song_dir)
        for i in range(len(pars)):
            par = pars[i]
            out_dir_par = jpath(out_song_dir, 'par_{}'.format(i + 1))
            create_if_not_exist(out_dir_par)
            par_path = jpath(out_dir_par, 'test.source')
            with open(par_path, 'w') as f:
                par = [line + '\n' for line in par]
                f.writelines(par)


def obtain_constraints():
    '''
    Obtain constraints for each of the paragraphs
    Target length: number of syllable in source text
    Target rhyme: rhy_0
    Target boundary: [bdr_0, bdr_0, ...]
    '''
    data_dir = '../../datasets/real'
    songs = ls(data_dir)
    for song in songs:
        song_path = jpath(data_dir, song)
        pars = ls(song_path)
        for par in pars:
            par_path = jpath(song_path, par)
            par_text_path = jpath(par_path, 'test.source')

            # Create dir for constraints
            p_folder = os.path.join(par_path, 'constraints')
            if not os.path.exists(p_folder):
                os.mkdir(p_folder)
            cons_path = os.path.join(p_folder, 'source')
            if not os.path.exists(cons_path):
                os.mkdir(cons_path)

            with open(par_text_path, 'r', encoding='utf8') as f:
                lines = f.readlines()

            # For length and rhyme constraints
            constrains = []
            t = len(lines)
            for i in tqdm(range(t)):
                line = lines[i]
                length = SyllableCounter.count_syllable_sentence(line)
                rhyme = 0
                constrains.append('{}\t{}'.format(length, rhyme))
            with open(os.path.join(cons_path, 'test.target'), 'w', encoding='utf-8') as f:
                f.writelines([i + '\n' for i in constrains])

            # For boundary constraints
            boundary_util = BoundaryUtilEn()
            constrains = []
            for line in tqdm(lines):
                t = line.strip()
                stress = boundary_util.sample_boundaries(t, n=0)
                constrains.append(stress)
            with open(os.path.join(cons_path, 'test_boundary.target'), 'w', encoding='utf-8') as f:
                f.writelines([i + '\n' for i in constrains])

def create_if_not_exist(p):
    if not os.path.exists(p):
        os.mkdir(p)


if __name__ == '__main__':
    _main()
