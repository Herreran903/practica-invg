% ==============================================
% model.mzn — Modelo CP disyuntivo para Job Shop (explicado)
% ----------------------------------------------------------
% Qué hace:
%   - Modela el JSP con variables de inicio por operación y restricciones:
%     • Precedencia dentro de cada job (orden de operaciones fijo).
%     • No solapamiento en cada máquina (una a la vez).
%     • Definición del makespan como máximo de (start+proc).
%   - Minimiza END_MAKESPAN.
%
% Entradas esperadas (desde .dzn):
%   JOBS, MACHINES
%   PROC_TIME[JOBS, MACHINES]       — duraciones por operación (I,J)
%   MACHINE_OF_OP[JOBS, MACHINES]   — id de máquina (1..M) por operación (I,J)
% ==============================================

% ---------- Dimensiones de la instancia ----------
int: JOBS;        % número de jobs
int: MACHINES;    % número de máquinas

% Conjuntos indexados por tamaño de la instancia
set of int: SET_JOBS = 1..JOBS;
set of int: SET_POS  = 1..MACHINES;  % posición/operación dentro de cada job

% ---------- Parámetros (datos) ----------
% Matriz de tiempos de proceso y asignación de máquina por operación (I,J)
array[SET_JOBS, SET_POS] of int: PROC_TIME;
array[SET_JOBS, SET_POS] of int: MACHINE_OF_OP;

% ---------- Aplanado de operaciones ----------
% Número total de operaciones y su índice lineal OPS = 1..JOBS*MACHINES
int: N_OPS = JOBS * MACHINES;
set of int: OPS = 1..N_OPS;

% Función que mapea (I,J) → posición lineal OP(I,J)
function int: OP(int: I, int: J) = (I-1) * MACHINES + J;

% Vectores planos de PT (duración) y MACH (máquina) en orden fila-major
array[OPS] of int: PT   = [ PROC_TIME[I,J]       | I in SET_JOBS, J in SET_POS ];
array[OPS] of int: MACH = [ MACHINE_OF_OP[I,J]   | I in SET_JOBS, J in SET_POS ];

% ---------- Horizonte y variables ----------
% Cota superior simple del horizonte (suma de todos los tiempos)
int: HORIZON = sum(I in SET_JOBS, J in SET_POS)(PROC_TIME[I,J]);

% S_FLAT[k] = tiempo de inicio de la operación k en 0..HORIZON
array[OPS] of var 0..HORIZON: S_FLAT;

% END_MAKESPAN = makespan a minimizar en 0..HORIZON
var 0..HORIZON: END_MAKESPAN;

% Acceso cómodo: S(I,J) = S_FLAT[OP(I,J)]
function var int: S(int: I, int: J) = S_FLAT[ OP(I,J) ];

% ---------- Restricciones ----------
% 1) Precedencia dentro del job:
%    La operación (I,J) debe terminar antes de que empiece (I,J+1)
constraint
  forall(I in SET_JOBS, J in 1..MACHINES-1) (
    S(I,J) + PROC_TIME[I,J] <= S(I, J+1)
  )
;

% 2) No solapamiento en máquinas:
%    Para dos operaciones A y B en la misma máquina, o A antes de B, o B antes de A.
predicate NO_OVERLAP(var int: S1, int: D1, var int: S2, int: D2) =
  S1 + D1 <= S2 \/ S2 + D2 <= S1;

constraint
  forall(A in OPS, B in OPS where A < B /\ MACH[A] = MACH[B]) (
    NO_OVERLAP(S_FLAT[A], PT[A], S_FLAT[B], PT[B])
  )
;

% 3) Definición del makespan:
%    END_MAKESPAN es al menos el fin de cada operación (start + duration)
constraint
  forall(I in SET_JOBS, J in SET_POS)(
    END_MAKESPAN >= S(I,J) + PROC_TIME[I,J]
  )
;

% ---------- Objetivo ----------
% Mantener esta sentencia exacta permite la inyección opcional de búsqueda desde Python.
solve minimize END_MAKESPAN;

% ---------- Salida legible ----------
% Imprime el makespan y, por filas (jobs), los tiempos de inicio S para cada operación.
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