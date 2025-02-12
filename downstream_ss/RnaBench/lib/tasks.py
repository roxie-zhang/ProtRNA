import numpy as np


class BaseTask():
    """
    Provides basic functionality of a Task.
    Takes keyword arguments and sets attributes.
    Provides functionality for using a task as a dictionary.
    """

    def __init__(self, **kwargs):
        for k, arg in kwargs.items():
            setattr(self, k, arg)

    def __getitem__(self, item):
        return getattr(self, item)

    def __len__(self):
        pass

    def __iter__(self):
        for att in vars(self):
            yield att

    def __repr__(self):
        return ', '.join([att for att in self])

    def items(self):
        for att in self:
            yield att, self[att]

    def keys(self):
        for att in self:
            yield att

    def to_torch(self):
        pass

    def to_numpy(self):
        pass

    def to_numeric(self):
        pass

    def to_dict(self):
        return {att: self[att] for att in self}


class RnaSequence():
    """
    A RNA sequence object.
    Provides attributes of a RNA sequence.
    """

    def __init__(self, seq_id, sequence, gc=None, length=None):
        """
        TODO
        """
        self._id = seq_id
        self._sequence = sequence
        # print(self._sequence)
        if gc is not None:
            self._gc = gc
        else:
            # print(sequence)
            self._gc = (''.join(sequence).upper().count('G') + ''.join(sequence).upper().count('C')) / len(
                ''.join(sequence))
        if length is not None:
            self.length = length
        else:
            self.length = len(sequence)

    def __len__(self):
        if isinstance(self.length, int) or isinstance(self.length, np.int64):
            return self.length
        else:
            return self.length[0]

    def __iter__(self):
        for s in self._sequence:
            yield s

    def to_numeric(self, stoi):
        """
        Change representaiton of sequence to integer.

        Input:
          stoi (dict): string to integer translation dict for sequence.
        Returns:
          New RnaSequence object.
        """
        new_seq = self.to_list()
        seq_id = new_seq.id
        seq = [stoi[s] for s in new_seq.sequence]
        gc = new_seq.gc
        return RnaSequence(seq_id=seq_id, sequence=seq, gc=gc, length=new_seq.length)

    def to_numpy(self, stoi):
        """
        Translate attributes to numpy arrays.
        """
        new_seq = self.to_numeric(stoi)
        seq_id = np.asarray(new_seq.id, dtype=np.int32)
        seq = np.asarray(new_seq.sequence, dtype=np.int32)
        gc = np.asarray(new_seq.gc, dtype=np.float64)
        length = np.asarray(new_seq.length, dtype=np.int32)
        return RnaSequence(seq_id=seq_id, sequence=seq, gc=gc, length=length)

    def to_list(self):
        """
        Translate sequence to list representation.
        """
        seq_id = [self._id]
        seq = self._sequence
        gc = [self._gc]
        length = [self.length]
        return RnaSequence(seq_id=seq_id, sequence=seq, gc=gc, length=length)

    @property
    def sequence(self):
        return self._sequence

    @property
    def gc(self):
        return self._gc

    @property
    def id(self):
        return self._id


class RnaStructure():
    """
    RNA structure object.
    Provides attributes of a RNA structure.
    """

    def __init__(
            self,
            struc_id,
            pairs,
            matrix=False,
            length=None,
    ):
        """
        TODO
        """
        self._id = struc_id
        self.pairs = pairs

        self.length = length

        if isinstance(matrix, bool) and matrix:
            self.prepare_matrix(length=length)
        else:
            self.matrix = matrix

    def __len__(self):
        if isinstance(self.length, list):
            return self.length[0]
        return self.length

    def __iter__(self):
        for pair in self.pairs:
            yield pair

    def prepare_matrix(self, length):
        """
        Prepare binary matrix representation from provided list of pairs.
        """
        self.matrix = np.zeros((length, length), dtype=np.int32)
        list(map(self.pair_to_matrix, self.pairs))

    def pair_to_matrix(self, pair):
        """
        Set one pair in matrix.
        """
        self.matrix[pair[0], pair[1]] = 1
        self.matrix[pair[1], pair[0]] = 1

    def to_numeric(self, stoi):
        """
        Translate structure to integer representation.

        Input:
          stoi (dict): string to integer translation dictionary for structure.

        Returns:
          new RnaStructure object.
        """
        new_struc = self.to_list()
        struc_id = new_struc.id
        pairs = self.pairs
        length = new_struc.length
        mat = new_struc.matrix
        return RnaStructure(
            struc_id=struc_id,
            pairs=pairs,
            length=length,
            matrix=mat,
        )

    def to_numpy(self, stoi):
        """
        Translate all attributes to numpy arrays
        """
        new_struc = self.to_numeric(stoi)
        struc_id = np.asarray(new_struc._id, dtype=np.int32)
        pairs = np.asarray(self.pairs)
        length = np.asarray(new_struc.length, dtype=np.int32)
        mat = new_struc.matrix
        return RnaStructure(
            struc_id=struc_id,
            pairs=pairs,
            matrix=mat,
            length=length,
        )

    def to_list(self):
        """
        Translate all attributes to list representation.
        """
        struc_id = [self._id]
        pairs = self.pairs
        length = [self.length]
        mat = self.matrix
        return RnaStructure(
            struc_id=struc_id,
            pairs=pairs,
            length=length,
            matrix=mat,
        )

    @property
    def num_pairs(self):
        return len(self.pairs)

    @property
    def structure(self):
        return pairs2db(self.pairs, self.length)

    @property
    def id(self):
        return self._id


class RNA():
    """
    Provides RNA objects.
    Contains a RnaSequence and RnaStructure object.
    Provides attributes of a RNA.
    """

    def __init__(self,
                 rna_id,
                 sequence,
                 structure=None,
                 pairs=None,
                 gc=None,
                 matrix=None,
                 length=None,
                 **kwargs,
                 ):
        if not isinstance(sequence, RnaSequence):
            self._sequence = RnaSequence(seq_id=rna_id, sequence=sequence, length=length, gc=gc)
        else:
            self._sequence = sequence
        if not isinstance(structure, RnaStructure):
            # assert pos1id is not None, f"No pairs pos1id provided for RNA {rna_id}"
            # assert pos2id is not None, f"No pairs pos2id provided for RNA {rna_id}"
            # assert pk is not None, f"No PK information provided for RNA {rna_id}"
            self._structure = RnaStructure(
                struc_id=rna_id,
                pairs=pairs,
                # seq_length=len(sequence),
                # pos1id=pos1id,
                # pos2id=pos2id,
                # pk=pk,
                matrix=matrix,
                length=length
                # energy=energy,
            )
        else:
            self._structure = structure
        self._id = rna_id
        for k, arg in kwargs.items():
            setattr(self, k, arg)

    def __len__(self):
        return len(self._sequence)

    def to_list(self):
        new_sequence = self._sequence.to_list()
        new_structure = self.structure.to_list()
        return RNA(
            rna_id=new_sequence.id,
            sequence=new_sequence,
            structure=new_structure
        )

    def to_numpy(self, seq_stoi, struc_stoi):
        new_sequence = self._sequence.to_numpy(seq_stoi)
        new_structure = self._structure.to_numpy(struc_stoi)
        return RNA(
            rna_id=new_sequence.id,
            sequence=new_sequence,
            structure=new_structure
        )

    def to_numeric(self, seq_stoi, struc_stoi):
        new_sequence = self._sequence.to_numeric(seq_stoi)
        new_structure = self._structure.to_numeric(struc_stoi)
        return RNA(
            rna_id=new_sequence.id,
            sequence=new_sequence,
            structure=new_structure
        )

    def to_dict(self):
        return {
          'id': self.id,
          'sequence': self.sequence,
          'pairs': self.structure.pairs,
          'gc_content': self.gc,
          'matrix': self._structure.matrix,
          'length': len(self),
        }

    def set_sequence(self, seq):
        self._sequence = RnaSequence(seq_id=self.id, sequence=seq, length=len(seq))

    def set_structure(self):
        self._structure = RnaStructure(
                                       struc_id=self._structure.id,
                                       pairs=self._structure.pairs,
                                       matrix=self._structure.matrix,
                                       length=len(self),
                                      )


    def prepare_matrix(self, length):
        if self._structure.matrix is not None:
            self._structure.prepare_matrix(length=length)


    @property
    def id(self):
        return self._id

    @property
    def sequence(self):
        return self._sequence.sequence

    @property
    def length(self):
        return len(self)

    @property
    def gc(self):
        return self._sequence.gc

    @property
    def num_pairs(self):
        return self._structure.num_pairs

    @property
    def matrix(self):
        return self._structure.matrix

    @property
    def pairs(self):
        return self._structure.pairs

