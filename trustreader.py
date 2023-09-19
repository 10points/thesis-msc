from surprise import Reader

class TrustReader(Reader):
    """The Reader class is used to parse a file containing trust values.
    Such a file is assumed to specify only one trust per line, and each line
    needs to respect the following structure: ::
        trustor ; trustee ; trust 
    where the order of the fields and the separator (here ';') may be
    arbitrarily defined (see below).  
    Args:
        line_format(:obj:`string`): The fields names, in the order at which
            they are encountered on a line. Please note that ``line_format`` is
            always space-separated (use the ``sep`` parameter). Default is
            ``'user item rating'``.
        sep(char): the separator between fields. Example : ``';'``.
        trust_scale(:obj:`tuple`, optional): The rating scale used for every
            trust values.  Default is ``(0, 1)``.
        skip_lines(:obj:`int`, optional): Number of lines to skip at the
            beginning of the file. Default is ``0``.
    """

    def __init__(
        self,
        line_format="trustor trustee trust",
        # sep=None,
        trust_scale=(0, 1),
        # skip_lines=0,
    ):

        
        # self.sep = sep
        # self.skip_lines = skip_lines
        super().__init__()
        self.trust_scale = trust_scale

        lower_bound, higher_bound = trust_scale

        splitted_format = line_format.split()

        entities = ["trustor", "trustee", "trust"]

        # check that all fields are correct
        if any(field not in entities for field in splitted_format):
            raise ValueError("line_format parameter is incorrect.")

        self.indexes = [splitted_format.index(entity) for entity in entities]

    def parse_line(self, line):
        """Parse a line.
        trusts are translated so that they are all strictly positive.
        Args:
            line(str): The line to parse
        Returns:
            tuple: trustor id, trustee id, trust .
        """

        line = line.split(self.sep)
        try:  
            trustor_id, trustee_id, t = (line[i].strip() for i in self.indexes)


        except IndexError:
            raise ValueError(
                "Impossible to parse line. Check the line_format" " and sep parameters."
            )

        return trustor_id, trustee_id, float(t)