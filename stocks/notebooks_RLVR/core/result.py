class Result:
    """
    A container for a successful operation's value or an error.
    Follows the "Railway Oriented Programming" pattern for error handling.
    """

    def __init__(self, val=None, err=None):
        self.val = val  # Holds the successful result
        self.err = err  # Holds the error message or exception

    @property
    def ok(self):
        """Returns True if the result is successful (no error)."""
        return self.err is None

    def __repr__(self):
        if self.ok:
            return f"Result(val={self.val!r})"
        else:
            return f"Result(err='{self.err}')"
