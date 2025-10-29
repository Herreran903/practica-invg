% ---------- Datos (idénticos a tu modelo) ----------
int: JOBS;
int: MACHINES;

set of int: SET_JOBS = 1..JOBS;
set of int: SET_POS  = 1..MACHINES;

array[SET_JOBS, SET_POS] of int: PROC_TIME;
array[SET_JOBS, SET_POS] of int: MACHINE_OF_OP;

int: N_OPS = JOBS * MACHINES;
set of int: OPS = 1..N_OPS;

function int: OP(int: I, int: J) = (I-1) * MACHINES + J;

array[OPS] of int: PT   = [ PROC_TIME[I,J]       | I in SET_JOBS, J in SET_POS ];
array[OPS] of int: MACH = [ MACHINE_OF_OP[I,J]   | I in SET_JOBS, J in SET_POS ];

% Cota superior del horizonte (Big-M razonable)
int: HORIZON = sum (I in SET_JOBS, J in SET_POS) (PROC_TIME[I,J]);
int: M = HORIZON;   % Big-M

% ---------- Variables ----------
array[OPS] of var 0..HORIZON: S_FLAT;     % tiempos de inicio
var 0..HORIZON: END_MAKESPAN;

function var int: S(int: I, int: J) = S_FLAT[ OP(I,J) ];

% Para cada par de operaciones en la MISMA máquina definimos 1 binaria de ORDEN:
% Y[i]=1 => (A antes que B), Y[i]=0 => (B antes que A)
set of int: IDX =
  1..sum(a in OPS, b in OPS where a < b /\ MACH[a] = MACH[b]) (1);

array[IDX] of int: PA = [ a | a in OPS, b in OPS where a < b /\ MACH[a] = MACH[b] ];
array[IDX] of int: PB = [ b | a in OPS, b in OPS where a < b /\ MACH[a] = MACH[b] ];

array[IDX] of var bool: Y;   % binarias de orden para pares en la misma máquina

% ---------- Restricciones ----------

% 1) Precedencia dentro de cada trabajo (misma que en tu CP)
constraint
  forall(I in SET_JOBS, J in 1..MACHINES-1) (
    S(I,J) + PROC_TIME[I,J] <= S(I, J+1)
  );

% 2) No solape en cada máquina (linealizado con Big-M y Y)
%    Si Y[i]=1: A antes B  ->  S_A + p_A <= S_B
%    Si Y[i]=0: B antes A  ->  S_B + p_B <= S_A
constraint
  forall(i in IDX) (
    S_FLAT[PA[i]] + PT[PA[i]] <= S_FLAT[PB[i]] + M * (1 - bool2int(Y[i])) /\
    S_FLAT[PB[i]] + PT[PB[i]] <= S_FLAT[PA[i]] + M *      bool2int(Y[i])
  );

% 3) Definición de makespan (lineal)
constraint
  forall(I in SET_JOBS, J in SET_POS) (
    END_MAKESPAN >= S(I,J) + PROC_TIME[I,J]
  );

% ---------- Objetivo ----------
solve minimize END_MAKESPAN;

% ---------- Salida ----------
output
[
  "END=", show(END_MAKESPAN), "\n",
  "S (START TIMES POR JOB):\n"
]
++
[
  concat([
    if J=1 then "" else " " endif ++
    show(S_FLAT[(I-1) * MACHINES + J])
  | J in SET_POS ])
  ++ "\n"
  | I in SET_JOBS
];
