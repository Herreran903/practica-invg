int: JOBS;
int: MACHINES;

set of int: SET_JOBS = 1..JOBS;
set of int: SET_POS = 1..MACHINES;

array[SET_JOBS, SET_POS] of int: PROC_TIME;
array[SET_JOBS, SET_POS] of int: MACHINE_OF_OP;

int: N_OPS = JOBS * MACHINES;
set of int: OPS = 1..N_OPS;

function int: OP(int: I, int: J) = (I-1) * MACHINES + J;

array[OPS] of int: PT = [ PROC_TIME[I,J] | I in SET_JOBS, J in SET_POS ];
array[OPS] of int: MACH = [ MACHINE_OF_OP[I,J] | I in SET_JOBS, J in SET_POS ];

int: HORIZON = sum(I in SET_JOBS, J in SET_POS)(PROC_TIME[I,J]);

array[OPS] of var 0..HORIZON: S_FLAT;
var 0..HORIZON: END_MAKESPAN;

function var int: S(int: I, int: J) = S_FLAT[ OP(I,J) ];

constraint
  forall(I in SET_JOBS, J in 1..MACHINES-1) (
    S(I,J) + PROC_TIME[I,J] <= S(I, J+1)
  )
;

predicate NO_OVERLAP(var int: S1, int: D1, var int: S2, int: D2) =
  S1 + D1 <= S2 \/ S2 + D2 <= S1;

constraint
  forall(A in OPS, B in OPS where A < B /\ MACH[A] = MACH[B]) (
    NO_OVERLAP(S_FLAT[A], PT[A], S_FLAT[B], PT[B])
  )
;

constraint
  forall(I in SET_JOBS, J in SET_POS)(
    END_MAKESPAN >= S(I,J) + PROC_TIME[I,J]
  )
;

solve minimize END_MAKESPAN;

output
[
  "END=", show(END_MAKESPAN), "\n",
  "S (START TIMES POR JOB):\n"
]
++
[
  concat([ 
    if J=1 then "" else " " endif ++ 
    show(S_FLAT[ (I-1) * MACHINES + J ]) 
  | J in SET_POS 
  ]) ++ "\n"
  | I in SET_JOBS
];