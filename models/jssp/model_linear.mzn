% ==============================================
% model_linear.mzn — Formulación lineal (MIP) del Job Shop (explicada)
% ---------------------------------------------------------------
% Propósito:
%   - Modelo lineal del JSP usando variables de inicio S y binarias Y para
%     decidir el orden relativo de pares de operaciones que comparten máquina.
%   - Las restricciones de no solape se linealizan con Big-M.
%   - Minimiza END_MAKESPAN.
%
% Entradas (desde .dzn):
%   JOBS, MACHINES
%   PROC_TIME[JOBS, MACHINES]       — duraciones por operación (I,J)
%   MACHINE_OF_OP[JOBS, MACHINES]   — id de máquina (1..M) por operación (I,J)
% ==============================================

% ---------- Datos y conjuntos ----------
int: JOBS;        % número de jobs
int: MACHINES;    % número de máquinas

set of int: SET_JOBS = 1..JOBS;
set of int: SET_POS  = 1..MACHINES;   % posición/operación dentro de cada job

% Tiempos de proceso y asignación de máquina por operación
array[SET_JOBS, SET_POS] of int: PROC_TIME;
array[SET_JOBS, SET_POS] of int: MACHINE_OF_OP;

% Aplanado de operaciones (índice lineal)
int: N_OPS = JOBS * MACHINES;
set of int: OPS = 1..N_OPS;

% Mapeo (I,J) → índice lineal OP(I,J) en 1..N_OPS
function int: OP(int: I, int: J) = (I-1) * MACHINES + J;

% Vectores planos por conveniencia
array[OPS] of int: PT   = [ PROC_TIME[I,J]       | I in SET_JOBS, J in SET_POS ];
array[OPS] of int: MACH = [ MACHINE_OF_OP[I,J]   | I in SET_JOBS, J in SET_POS ];

% ---------- Horizonte y Big-M ----------
% Cota superior del horizonte: suma de todas las duraciones.
int: HORIZON = sum (I in SET_JOBS, J in SET_POS) (PROC_TIME[I,J]);
% Constante Big-M. Usamos HORIZON como cota holgada, suficiente para garantizar
% que la desigualdad "se apaga" cuando la binaria cambia de valor.
int: M = HORIZON;

% ---------- Variables ----------
% S_FLAT[k] = tiempo de inicio de la operación k
array[OPS] of var 0..HORIZON: S_FLAT;
% Makespan
var 0..HORIZON: END_MAKESPAN;

% Acceso helper: S(I,J) = S_FLAT[OP(I,J)]
function var int: S(int: I, int: J) = S_FLAT[ OP(I,J) ];

% Para cada par de operaciones en la MISMA máquina definimos 1 binaria de ORDEN:
% Y[i] = 1  ⇒  A antes que B
% Y[i] = 0  ⇒  B antes que A
set of int: IDX =
  1..sum(a in OPS, b in OPS where a < b /\ MACH[a] = MACH[b]) (1);

% Listas alineadas de pares (A,B) por cada índice i
array[IDX] of int: PA = [ a | a in OPS, b in OPS where a < b /\ MACH[a] = MACH[b] ];
array[IDX] of int: PB = [ b | a in OPS, b in OPS where a < b /\ MACH[a] = MACH[b] ];

% Binarias de orden para los pares en la misma máquina
array[IDX] of var bool: Y;

% ---------- Restricciones ----------
% 1) Precedencia dentro de cada job (misma que en el modelo CP):
%    Operación (I,J) debe terminar antes de que empiece (I,J+1)
constraint
  forall(I in SET_JOBS, J in 1..MACHINES-1) (
    S(I,J) + PROC_TIME[I,J] <= S(I, J+1)
  );

% 2) No solape en cada máquina (linealizado con Big-M y Y):
%    - Si Y[i] = 1, entonces A antes que B:
%         S_A + p_A <= S_B
%      Se implementa como:
%         S_A + p_A <= S_B + M*(1 - Y[i])   (activa cuando Y[i]=1)
%    - Si Y[i] = 0, entonces B antes que A:
%         S_B + p_B <= S_A
%      Se implementa como:
%         S_B + p_B <= S_A + M*(Y[i])       (activa cuando Y[i]=0)
constraint
  forall(i in IDX) (
    S_FLAT[PA[i]] + PT[PA[i]] <= S_FLAT[PB[i]] + M * (1 - bool2int(Y[i])) /\
    S_FLAT[PB[i]] + PT[PB[i]] <= S_FLAT[PA[i]] + M *      bool2int(Y[i])
  );

% 3) Definición lineal del makespan:
%    END_MAKESPAN ≥ S(I,J) + PROC_TIME[I,J]  para toda operación
constraint
  forall(I in SET_JOBS, J in SET_POS) (
    END_MAKESPAN >= S(I,J) + PROC_TIME[I,J]
  );

% ---------- Objetivo ----------
% Importante: mantener exactamente esta línea facilita inyección de estrategia
% desde Python en el modelo CP. Aquí, en MIP, no se inyecta búsqueda.
solve minimize END_MAKESPAN;

% ---------- Salida ----------
% Imprime makespan y, por job, los tiempos de inicio de cada operación.
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
