"""nr_burst_finder

Ever tried to automatically identify the data bursts or patterns in the collected slot measurements?

The ``nr_burst_finder`` module is an example on how to do this using numpy and scipy.
Since the slot measurements may contain gaps i.e. the measurements may not be reported for every consecutive slot,
first the reports are mapped onto the full time slot grip table.

The burst finding method consists of finding cumulative sum of consecutive values using a sliding window of a given length in slots.
For that purpose a discrete, linear convolution with a sequence of ones (mask) of a given sliding window length is used (ref. ``np.convolve``).
Afterwards the peaks with the height of at least the given burst size are located using the ``scipy.signal.find_peaks``
which provide the indices of the beginning of bursts in the full time slot grip table as well as the bursts' sizes.

The pattern finding method consists of converting the full time slot grip table into the 2d array with number of rows equal
to the number of elements in the searched pattern and number of columns equal to
``len(full time slot grip table) - len(pattern) + 1`` where each row is rolled by 1 element to the left.
This way each column represents consecutive len(pattern) elements of the input array starting from the element with
an index equal to the column index and then matching the columns with the flipped pattern array. The indices of the matched columns
indicate the start position of the pattern found in the input array.

This method uses the numpy pretty cool concept of ``strides`` (i.e. shifting in both dimensions by byte length of a single element).
Ref. ``numpy.lib.stride_tricks.as_strided``.

For the example usage check out the ``nr_burst_finder.tbs_test`` function or simply execute the script. Here is how to execute it::

    >>> from nr_burst_finder import nr_burst_finder
    >>> nr_burst_finder.main()
    INFO:nr_burst_finder.nr_burst_finder:Collected measurements:
      SFN, slot, TBS
    [[  29   18    0]
     [  30    8  500]
     [  30    9 1000]
     [  30   18  500]
     [  31    8    0]
     [  31    9    0]
     [  31   18    0]
     [  32    8  500]
     [  32    9 1000]
     [  32   18  200]
     [  33    8    0]
     [  33    9 1000]
     [  33   18    0]
     [  33   19    0]
     [  34    8    0]
     [  34    9    0]
     [  34   18    0]
     [  35    8    0]
     [  35    9    0]
     [  35   18    0]
     [  35   19  250]
     [  36    8  250]
     [  36    9 1000]
     [  36   18  500]
     [  36   19    0]
     [  37    8    0]
     [  37    9    0]
     [  37   18    0]
     [  37   19    0]
     [  38    8    0]
     [  38    9    0]
     [  38   18    0]
     [  38   19    0]
     [  39    8    0]
     [  39    9 1000]
     [  39   18 2000]]
    INFO:nr_burst_finder.nr_burst_finder:Verifying bursts (burst_min_size:2000, burst_window:20, range of expected bursts:(3, 4))
    INFO:nr_burst_finder.nr_burst_finder:Checking SAMPLE TBS bursts (min. burst size:2000, burst window: 20 )
    INFO:nr_burst_finder.nr_burst_finder:Mapping SAMPLE TBS reports to full time range table (20 slots margins prepended to the beginning and appended at the end)
    INFO:nr_burst_finder.nr_burst_finder:Expected (3, min:3, max:4) SAMPLE TBS bursts>=2000 detected in (sfn/slot/size): [[30.0, 3.0, 2000.0], [35.0, 19.0, 2000.0], [39.0, 4.0, 3000.0]](measurement time: 29/18 - 39/18)
    INFO:nr_burst_finder.nr_burst_finder:Verifying TBS pattern
    INFO:nr_burst_finder.nr_burst_finder:Checking SAMPLE TBS pattern in consecutive slots(pattern:[0, 0, 0, 0, 500, 1000], compare_values:True)
    INFO:nr_burst_finder.nr_burst_finder:Mapping SAMPLE TBS reports to full time range table (0 slots margins prepended to the beginning and appended at the end)
    INFO:nr_burst_finder.nr_burst_finder:SAMPLE TBS pattern detected 2 times in (sfn/slot): [[30, 4], [32, 4]](measurement time: 29/18 - 39/18)


"""
from __future__ import annotations
from typing import List, Union, Tuple, Optional
import logging
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import signal

logger = logging.getLogger(__name__)


class FinderBase:
    """Base class for bursts and patterns finder

    """
    def __init__(self, arr: np.array, add_info: str = ""):
        self._channel_name = ""
        self._quantity = ""
        self._add_info = add_info
        self._arr: np.array = arr

    @property
    def verified_quantity(self) -> str:
        _str = f"{self._channel_name} {self._quantity}"
        if self._add_info:
            _str = f"{_str} ({self._add_info})"
        return _str

    @property
    def values(self) -> np.array:
        return self._arr[:, 2]

    @property
    def time(self) -> np.array:
        return self._arr[:, (0, 1)]

    @property
    def arr(self) -> np.array:
        return self._arr

    @property
    def start_time(self) -> str:
        return f"{self.time[0][0]}/{self.time[0][1]}" if len(self.time) else "N/A"

    @property
    def end_time(self) -> str:
        return f"{self.time[-1][0]}/{self.time[-1][1]}" if len(self.time) else "N/A"

    def min_inx(self) -> np.array:
        return np.argmin(self.values)

    def max_inx(self) -> np.array:
        return np.argmax(self.values)

    def check_valid(self) -> bool:
        if self.values.size < 2:
            logger.warning(f"{self.verified_quantity} measurements insufficient "
                           f"(numOfSamples {self._arr.size})")
            return False
        return True

    def in_full_time_span(self, slots_in_subframe: int = 2,
                          extra_slots: int = 0,
                          default_val: Optional[Union[int, float, np.nan]] = 0) -> np.array:
        """Fills the collected reports in a full time range 2d array with number of rows calculated
        as a difference between slot time of a first and last report.

        Args:
            slots_in_subframe: number of slots per subframe (depends on the cell's numerology)
            extra_slots: number of slots to prepend to the beginning and append to the end of the generated time span
            default_val: default value to be put in empty cells

        Returns:
            a 2d array with number of rows equal to a difference between slot time of a first and last report.
        """
        _slots_in_sfn = 10 * slots_in_subframe
        _in_full_time = None
        try:
            logger.info(f"Mapping {self.verified_quantity} reports to full time range table "
                        f"({extra_slots} slots margins prepended to the beginning and appended at the end)")
            # array with reports times is slot unit (SFN*numOfSlotinSFN + slot)
            _rep_time_in_slots = self._arr[:, 0] * _slots_in_sfn + self._arr[:, 1]
            # array with all slots >= first report time and <= last report time
            _full_time_in_slots = np.arange(_rep_time_in_slots[0] - extra_slots,
                                            _rep_time_in_slots[-1] + 1 + extra_slots)
            # determine the shape of a full_time_span column stack
            # (n_rows== number of slots between first and last report slot,
            # n_cols - number of columns in the column stack)
            _shape = (len(_full_time_in_slots), self._arr.shape[1])
            # prepare an empty (filled with a default_val) 2d array with a given shape
            _in_full_time = np.empty(_shape, dtype='int' if isinstance(default_val, int) else 'float')
            _in_full_time.fill(default_val)
            # fill first two columns with {SFN, slot} pair calculated from _full_time_in_slots values
            _in_full_time[:, 0] = (_full_time_in_slots / 20).astype('int')
            _in_full_time[:, 1] = (_full_time_in_slots % 20).astype('int')
            # find rows in the new array for which an {SFN, slot} pairs intersect
            # with {SFN, slot} pairs of the collected reports
            _rows_with_matching_time = np.isin(_full_time_in_slots, _rep_time_in_slots)
            # merge collected reports to the new full_time_span array by assigning them to rows with matching {SFN, slot} pair
            _in_full_time[_rows_with_matching_time] = self._arr
        except Exception as e:
            logger.warning(f"Could not map {self.verified_quantity} reports over time ({repr(e)})")
        return _in_full_time

    def check_bursts(self, min_size: int, window: int,
                     num_of_bursts: Optional[Tuple[int, int]] = (1,),
                     slots_in_subframe: Optional[int] = 2):
        """Checks if measured quantity bursts of a given height and within a
        given period can be identified in the collected reports

        Args:
            min_size: minimum burst size, accumulated measured quantity over a given period
            window:  number of slots over which to accumulate the measured quantity to find peaks
            num_of_bursts: tuple with min and max number of bursts to expect,
                           if single element tuple only check against min
            slots_in_subframe: number of slots per subframe (depends on the cell's numerology)

        """
        logger.info(f"Checking {self.verified_quantity} bursts "
                    f"(min. burst size:{min_size}, burst window: {window} )")
        msg = f"{self.verified_quantity} bursts>={min_size}"
        # fill the measured quantity in the time array with a slot grid starting `window` slots before the first
        # and ending `window` slots after the last occurrence of the measured quantity,
        # time slots without the measurement are filled with zeros
        _input_arr = self.in_full_time_span(slots_in_subframe=slots_in_subframe, extra_slots=window)
        if _input_arr is not None:
            # using a sliding window of slot length given by a `window` find accumulated values
            # (for that purpose a discrete, linear convolution with a sequence of ones (mask)
            # using only completely overlapping points i.e. mode='valid')
            # alternatively the following could be used:
            #from numpy.lib.stride_tricks import sliding_window_view
            #_cummulative_sum = np.sum(sliding_window_view(_input_arr[:, -1], window_shape=window), axis=1)
            _cummulative_sum = np.convolve(_input_arr[:, -1], np.ones(window, dtype='int'), 'valid')
            # find peaks with minimum height of min_size
            _peaks = signal.find_peaks(_cummulative_sum, height=min_size)
            _occurences = _peaks[0].size
            if _occurences < num_of_bursts[0]:
                logger.warning(f"Insufficient {msg} detected: {_occurences} < minimum target:{num_of_bursts[0]} "
                               f"(measurement time: {self.start_time} - {self.end_time})")
            elif len(num_of_bursts) == 2 and _occurences > num_of_bursts[1]:
                logger.warning(f"Too many {msg} detected: {_occurences} > maximum target:{num_of_bursts[1]} "
                               f"(measurement time: {self.start_time} - {self.end_time})")
            else:
                # find their location in the _input_arr and then get corresponding sfn, slot pair
                # together with the corresponding peak height
                _res =np.c_[_input_arr[np.ix_(_peaks[0], [0, 1])], _peaks[1].get('peak_heights')]
                _msg_prefix = f"Expected ({_occurences}, min:{num_of_bursts[0]}, " \
                              f"max:{'n/a' if len(num_of_bursts) < 2 else num_of_bursts[-1]})"
                logger.info(f"{_msg_prefix} {msg} detected in (sfn/slot/size): {_res.tolist()}"
                            f"(measurement time: {self.start_time} - {self.end_time})")
        else:
            logger.warning(f"{msg} not found "
                           f"(No or insufficient measurement data, "
                           f"measurement time: {self.start_time} - {self.end_time})")

    def check_pattern(self, pattern: List[int], slots_in_subframe: Optional[int] = 2, compare_values: bool = True):
        """Checks if pattern of values mapped to the consecutive slots can be found in the collected measurements

        Note. Only 1d patterns supported

        Args:
            pattern: pattern of values mapped to consecutive slots to search for in the collected measurements
            slots_in_subframe: number of slots per subframe (depends on the cell's numerology)
            compare_values: if True, values are also compared, otherwise just binary check for value presence (1) or not (0)
        """
        logger.info(f"Checking {self.verified_quantity} pattern in consecutive slots"
                    f"(pattern:{pattern}, compare_values:{compare_values})")
        msg = f"{self.verified_quantity} pattern"
        # fill the measured quantity in the time array with a slot grid
        # time slots without the measurement are filled with zeros
        _input_arr = self.in_full_time_span(slots_in_subframe=slots_in_subframe)
        _pattern_size = len(pattern)
        if _input_arr is not None and 0 < _pattern_size < _input_arr[:, -1].size:
            # depending on the ``check_values`` boolean either use the original data type or boolean
            # for the pattern matching
            _vals = np.array(_input_arr[:, -1], dtype=_input_arr.dtype if compare_values else bool)
            # from the array of measured quantity in full slot grid create 2d array
            # with number of rows equal to number of elements in the searched pattern
            # and number of columns equal to len(_input_vals) - len(pattern) + 1
            # where each row is rolled by 1 element to the left.
            # This way each column represents consecutive len(pattern) elements of the input array,
            # starting from the element with an index equal to the column index
            # Note. for this purpose the numpy concept of ``strides`` is used (shifting in both dimensions by byte length
            # of a single element)
            _stride = _vals.strides[0]
            _roll_arr = as_strided(_vals,
                                   shape=(_pattern_size, _vals.size - _pattern_size + 1),
                                   strides=(_stride, _stride)
                                   )
            # flip the pattern to be single column multi-row array
            _pattern_flipped = np.array(pattern, dtype=_vals.dtype)[:, np.newaxis]
            # find all columns matching the flipped pattern and return their indices.
            # These indices indicate the start position of the pattern found in the input array
            _occurrences = np.where(np.all(_roll_arr == _pattern_flipped, axis=0))[0]
            if _occurrences.size:
                _res = _input_arr[np.ix_(_occurrences, [0, 1])]
                logger.info(f"{msg} detected {_occurrences.size} times "
                            f"in (sfn/slot): {_res.tolist()}"
                            f"(measurement time: {self.start_time} - {self.end_time})")
            else:
                logger.warning(f"{msg} not found "
                               f"(measurement time: {self.start_time} - {self.end_time})")
        else:
            logger.warning(f"{msg} not found "
                           f"(No or insufficient measurement data, "
                           f"measurement time: {self.start_time} - {self.end_time})")


class TbsBurstFinder(FinderBase):
    """Example class for TransportBlockSize bursts and patterns finder

    """
    def __init__(self, tbs_arr: np.array, add_info: str = ""):
        super().__init__(arr=tbs_arr, add_info=add_info)
        self._quantity = "TBS"
        self._channel_name = "SAMPLE"


def tbs_test():
    """Example function to showcase how to use the module

    For the defined Transport Block Size measurements collection identify the bursts of a given size and duration
    and find the patterns in the full slot grid table.

    """
    tbs_arr = [
           [29, 18,    0],
           [30,  8,  500],
           [30,  9, 1000],
           [30, 18,  500],
           [31,  8,    0],
           [31,  9,    0],
           [31, 18,    0],
           [32,  8,  500],
           [32,  9, 1000],
           [32, 18,  200],
           [33,  8,    0],
           [33,  9, 1000],
           [33, 18,    0],
           [33, 19,    0],
           [34,  8,    0],
           [34,  9,    0],
           [34, 18,    0],
           [35,  8,    0],
           [35,  9,    0],
           [35, 18,    0],
           [35, 19,  250],
           [36,  8,  250],
           [36,  9, 1000],
           [36, 18,  500],
           [36, 19,    0],
           [37,  8,    0],
           [37,  9,    0],
           [37, 18,    0],
           [37, 19,    0],
           [38,  8,    0],
           [38,  9,    0],
           [38, 18,    0],
           [38, 19,    0],
           [39,  8,    0],
           [39,  9, 1000],
           [39, 18, 2000]
    ]
    verifier = TbsBurstFinder(tbs_arr=np.array(tbs_arr))
    logger.info(f"Collected measurements:\n  SFN, slot, TBS\n{verifier.arr}")
    burst_min_size = 2000
    burst_window = 20
    burst_number = (3, 4)
    logger.info(f"Verifying bursts (burst_min_size:{burst_min_size}, burst_window:{burst_window}, "
                f"range of expected bursts:{burst_number})")
    verifier.check_bursts(min_size=burst_min_size, window=burst_window, num_of_bursts=burst_number)
    val_pattern = [0, 0, 0, 0, 500, 1000]
    logger.info(f"Verifying TBS pattern")
    verifier.check_pattern(pattern=val_pattern)


def main():
    logging.basicConfig(level=logging.INFO)
    tbs_test()


if __name__ == "__main__":
    main()
