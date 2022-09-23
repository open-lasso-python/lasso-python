
Performance Info
----------------

---------------


D3plot
~~~~~~

    Benchmark:
        The d3plot reader is bazingly fast by using several memory tricks.
        While postprocessors build up an internal datastructure during reading,
        this reader avoids this and simply references memory within the files. 
        In consequence performance benchmarks show that the runtime
        of the code is solely dominated by pulling the files into memory.

        ::

            2108 function calls (2099 primitive calls) in 43.017 seconds 

            Ordered by: internal time

            ncalls  tottime  percall  cumtime  percall filename:lineno(function)
                35   38.960    1.113   38.960    1.113 {method 'readinto' of '_io.BufferedReader' objects}
                35    2.366    0.068    2.366    0.068 {built-in method io.open}
                1     1.644    1.644   42.959   42.959 D3plot.py:2565(_read_state_bytebuffer)
                71    0.043    0.001    0.043    0.001 {built-in method nt.stat}
                2     0.002    0.001    0.057    0.028 BinaryBuffer.py:234(load)
                70    0.000    0.000    0.001    0.000 ntpath.py:74(join)
                142   0.000    0.000    0.000    0.000 ntpath.py:121(splitdrive)
                1     0.000    0.000    0.019    0.019 D3plot.py:2738(<listcomp>)
                1     0.000    0.000    0.000    0.000 {built-in method nt.listdir}
                36    0.000    0.000    0.000    0.000 {method 'match' of '_sre.SRE_Pattern' objects}
                84    0.000    0.000    0.000    0.000 {built-in method numpy.core.multiarray.frombuffer} 
                1     0.000    0.000   42.959   42.959 D3plot.py:1304(_read_states)
                ...

        In the table above the largest, first three performance issues are all related
        to loading files into memory, accounting for 99.89% of runtime. 
        The routines reading the data arrays (deserialization) have an almost constant 
        runtime of a few milliseconds and are independent of the filesize. 
    
    Efficiency:
        Note that writing such efficient and simple code in C or any other language 
        is much more challenging than in Python, thus surprisingly this implementation 
        can be expected to outperform most native codes and tools. Indeed tests show that
        when reading all results of a d3plot in a postprocessor, this library is 
        devastatingly fast. For reading single node-fields this library can be slower
        though (see Improvements below).

    Array-based API:
        Building objects is useful (e.g. neighbor search) but requires additional
        runtime overhead. By the principle of "don't pay for what you don't use"
        it was decided to avoid object orientation, thus providing a majorly 
        array-based API. This is perfectly suitable for data analysis as well 
        as machine learning.

    Improvements:
        A speedup 'could' be achieved by using memory-mapping, thus pulling only
        fractions of the files into the RAM which hold the required arrays. 
        Postprocessors sometimes do this when reading certain fields, outperforming
        this library in such a case (e.g. node displacements). Since dyna litters 
        most arrays across the entire file though, this method was not considered 
        worth the effort and thus this library always reads everything.


Binout
~~~~~~

    The binout is used to store arbitrary binary data at a much higher 
    frequency than the d3plot. As a result the data is dumped in an internal
    'state folder'. Since different results may be dumped at different 
    frequencies some state folders might contain more information than 
    others. 
    This inherently prevents efficient memory reading. The python version used
    here is slower than the original C-based version but one therefore gains
    better portability accross operating systems.

 