
# Nimble Quickstart

Nimble is package to find and filter out events that meet user specified, multivariate criteria. Once the events are found, it can then return a mask, a Numpy array representation, or a Pandas series representation.


```python
import numpy as np
from nimble import Events
import matplotlib.pyplot as plt
%matplotlib inline
```

## Getting Started

First, create some fake time series data with a sample period of 1 second.


```python
np.random.seed(2)
x = np.random.randint(0, 2, 25)
y = np.random.randint(2, 5, 25)
sample_period = 1
index = list(range(0, 25))
```


```python
def plot():
    # Convenience function to plot events vs condition
    plt.plot(index, x, label='condition')
    plt.plot(events.as_array(), label='events', linestyle='--', color='r')
    plt.yticks([-1,0,1,2])
    plt.xlabel('Time(s)')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks(index)
    plt.grid()
    plt.show()
```


```python
plt.plot(index, x)
plt.yticks([-1,0,1,2])
plt.xlabel('Time(s)')
plt.ylabel('Value')
plt.xticks(index)
plt.grid()
plt.show()
```


![png](output_5_0.png)


## Basic Usage

First, find the events where x>0. In this case, the events and the condition will be identical arrays.


```python
events = Events(x>0, sample_period=1).find()
```


```python
plot()
```


![png](output_8_0.png)


## Debouncing

Debouncing prevents fast cycling from activating or deactivating events. More information on debouncing can be found [here](https://en.wikipedia.org/wiki/Switch#Contact_bounce)

> The effect is usually unimportant in power circuits, but causes problems in some analogue and logic circuits that respond fast enough to misinterpret the on‑off pulses as a data stream. (Wikipedia)

Debounce parameters are specified in the number of consecutive seconds required for the condition to be true or false in order to activate or deactivate an event. Both parameters are inclusive. So if `activation_debounce=4`, then the condition must be true for greater than or equal to 4 seconds to activate an event. If `deactivation_debounce=2`, then the condition will need to be `False` for greater than or equal to 2 seconds in order for an activated event to deactivate.


```python
events = Events(x>0, sample_period=1, activation_debounce=1, deactivation_debounce=2).find()
```


```python
plot()
```


![png](output_11_0.png)


## Event Duration Filtering

Event's can be filtered out by both their minimum and maximum durations. Again, these parameters are inclusive. So if `min_duration=3`, then any event greater than or equal to 3 seconds will not be excluded. If `max_duration=5`, then any event less than or equal to 5 seconds in duration will not be excluded


```python
events = Events(x>0, sample_period=1, min_duration=3, max_duration=5).find()
```


```python
plot()
```


![png](output_14_0.png)


## Offsetting Event Start and Stop Values




```python
events = Events(x>0, sample_period=1, deactivation_debounce=2, min_duration=3, start_offset=-1, stop_offset=1).find()
```


```python
plot()
```


![png](output_17_0.png)


## A Note on Execution Order and the `find()` Method


```python
plt.subplot(3,1,1)
plt.plot(index, x, label='x')
plt.ylabel('x')
plt.yticks([-1,0,1,2])
plt.xticks(index)
plt.grid()

plt.subplot(3,1,2)
plt.plot(index, y, label='y')
plt.ylabel('y')
plt.yticks(list(range(6)))
plt.xticks(index)
plt.grid()

plt.subplot(3,1,3)
plt.plot(events.as_array(), label='trigger', linestyle='--', color='r')
plt.yticks([-1,0,1,2])
plt.xlabel('Time(s)')
plt.ylabel('trigger')
plt.xticks(index)
plt.grid()

plt.show()
    

```


![png](output_19_0.png)


## Iterating Over Events




```python
string='For event {}, y at the start is {} and top is {}'
for event in events:
    print('{} is the average of y during event {}'.format(np.mean(x[event.islice]), event.i))
    
    print(string.format(event.i, y[event.istart], y[event.istop]))
```

    0.6666666666666666 is the average of y during event 0
    For event 0, y at the start is 4 and top is 2
    0.75 is the average of y during event 1
    For event 1, y at the start is 4 and top is 2
    

## Special Methods of `Events`

The function `len(events)` tells you how many events were found


```python
len(events)
```




    2



You can get a quick summary of all of the events by doing `print()`


```python
print(events)
```

    Number of events: 2
    Min, Max, Mean Duration: 4.000s, 15.000s, 9.500s
    sample_period: 1s,
    activation_debounce: None, deactivation_debounce: 2s,
    min_duration: 3s, max_duration: None,
    start_offset: -1s, stop_offset: 1s
    

## Multivariate Conditions




```python
events = Events((x>0) & (y>3), sample_period=1, deactivation_debounce=2, min_duration=3, start_offset=-1, stop_offset=1).find()
```


```python
plt.subplot(3,1,1)
plt.plot(index, x, label='x')
plt.ylabel('x')
plt.yticks([-1,0,1,2])
plt.xticks(index)
plt.grid()

plt.subplot(3,1,2)
plt.plot(index, y, label='y')
plt.ylabel('y')
plt.yticks(list(range(6)))
plt.xticks(index)
plt.grid()

plt.subplot(3,1,3)
plt.plot(events.as_array(), label='trigger', linestyle='--', color='r')
plt.yticks([-1,0,1,2])
plt.xlabel('Time(s)')
plt.ylabel('trigger')
plt.xticks(index)
plt.grid()

plt.show()
```


![png](output_28_0.png)

