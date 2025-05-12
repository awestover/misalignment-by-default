- intro
  - summary of exps

Q1: Do models consistently become misaligned when we tune on generic data?
E1: Ideally we'd have a nice set of experiments of the form {dataset} X {score on several personality tests}

Q2: How are the goals changing?

E2: Ideally we'd show: 

- The model isn't just getting generically more evil.
- It's getting more obedient, and more power seeking, and
    alignment fakey.
- We'd support the second claim with COT traces, ideally graded COT traces.

Q3: Why are the values changing?

E3: Ideally we'd show: 

- base model + safety trained model converge to similar values on misalignment datasets, and on Hendricks' utility engineering datasets. 

Q4: How good of an analogy is this?

E4: We maybe see some long run drift.

-----

## Experiments that you should run:

* Ft a big guy and check how their AFing propensity changes.

