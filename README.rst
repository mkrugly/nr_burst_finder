nr_burst_finder
===============

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

Usage
-----

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
