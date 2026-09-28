/**
 * A small arithmetic language for the scalar field calculator, e.g.
 * `abs([C2C distance]) * 1000` or `intensity > 200 and z < 5`. It is parsed,
 * never evaluated as JavaScript, so expressions from sessions are safe.
 *
 * Grammar (lowest precedence first): `or`, `and`, comparisons
 * (< <= > >= == !=, giving 1 or 0), + -, * / %, unary - and `not`, ^ (right
 * associative), then numbers, fields (a bare name or `[any name]`),
 * functions and parentheses.
 */

type Node =
  | { kind: "number"; value: number }
  | { kind: "field"; name: string }
  | { kind: "unary"; op: "-" | "not"; arg: Node }
  | { kind: "binary"; op: string; left: Node; right: Node }
  | { kind: "call"; name: string; args: Node[] };

const FUNCTIONS: Record<string, { arity: number; fn: (...a: number[]) => number }> = {
  abs: { arity: 1, fn: Math.abs },
  sqrt: { arity: 1, fn: Math.sqrt },
  exp: { arity: 1, fn: Math.exp },
  log: { arity: 1, fn: Math.log },
  log10: { arity: 1, fn: Math.log10 },
  sin: { arity: 1, fn: Math.sin },
  cos: { arity: 1, fn: Math.cos },
  tan: { arity: 1, fn: Math.tan },
  floor: { arity: 1, fn: Math.floor },
  ceil: { arity: 1, fn: Math.ceil },
  round: { arity: 1, fn: Math.round },
  min: { arity: 2, fn: Math.min },
  max: { arity: 2, fn: Math.max },
  atan2: { arity: 2, fn: Math.atan2 },
  clamp: { arity: 3, fn: (v, lo, hi) => Math.min(hi, Math.max(lo, v)) },
  if: { arity: 3, fn: (c, a, b) => (c ? a : b) },
};

const CONSTANTS: Record<string, number> = { pi: Math.PI, e: Math.E, nan: Number.NaN };

type Token = { kind: "num"; value: number } | { kind: "name"; value: string } | { kind: "op"; value: string };

function tokenize(text: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < text.length) {
    const c = text[i];
    if (/\s/.test(c)) {
      i++;
    } else if (/[0-9.]/.test(c)) {
      const m = /^(\d+\.?\d*|\.\d+)(e[+-]?\d+)?/i.exec(text.slice(i));
      if (!m) throw new Error(`bad number at ${i + 1}`);
      tokens.push({ kind: "num", value: Number(m[0]) });
      i += m[0].length;
    } else if (c === "[") {
      const end = text.indexOf("]", i);
      if (end < 0) throw new Error("missing ]");
      tokens.push({ kind: "name", value: text.slice(i + 1, end).trim() });
      i = end + 1;
    } else if (/[A-Za-z_]/.test(c)) {
      const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(text.slice(i))!;
      tokens.push({ kind: "name", value: m[0] });
      i += m[0].length;
    } else {
      const two = text.slice(i, i + 2);
      if (["<=", ">=", "==", "!=", "&&", "||"].includes(two)) {
        tokens.push({ kind: "op", value: two === "&&" ? "and" : two === "||" ? "or" : two });
        i += 2;
      } else if ("+-*/%^()<>,!".includes(c)) {
        tokens.push({ kind: "op", value: c === "!" ? "not" : c });
        i++;
      } else {
        throw new Error(`unexpected "${c}" at ${i + 1}`);
      }
    }
  }
  return tokens;
}

/** Parse an expression; throws with a readable message on errors. */
export function parse(text: string): Node {
  const tokens = tokenize(text);
  let pos = 0;
  const peek = () => tokens[pos];
  const isOp = (...ops: string[]) => {
    const t = peek();
    return !!t && (t.kind === "op" || t.kind === "name") && ops.includes(t.value);
  };
  const expect = (op: string) => {
    if (!isOp(op)) throw new Error(`expected "${op}"`);
    pos++;
  };
  const binary = (next: () => Node, ops: string[]) => (): Node => {
    let left = next();
    while (isOp(...ops)) {
      const op = String(tokens[pos++].value);
      left = { kind: "binary", op, left, right: next() };
    }
    return left;
  };
  const primary = (): Node => {
    const t = tokens[pos++];
    if (!t) throw new Error("unexpected end");
    if (t.kind === "num") return { kind: "number", value: t.value };
    if (t.kind === "op" && t.value === "(") {
      const inner = or();
      expect(")");
      return inner;
    }
    if (t.kind === "name") {
      const name = t.value;
      const fn = FUNCTIONS[name.toLowerCase()];
      if (fn && isOp("(")) {
        pos++;
        const args: Node[] = [];
        if (!isOp(")")) {
          args.push(or());
          while (isOp(",")) {
            pos++;
            args.push(or());
          }
        }
        expect(")");
        if (args.length !== fn.arity) throw new Error(`${name} takes ${fn.arity} argument${fn.arity > 1 ? "s" : ""}`);
        return { kind: "call", name: name.toLowerCase(), args };
      }
      if (name.toLowerCase() in CONSTANTS) return { kind: "number", value: CONSTANTS[name.toLowerCase()] };
      return { kind: "field", name };
    }
    throw new Error(`unexpected "${t.value}"`);
  };
  const power = (): Node => {
    const base = primary();
    if (isOp("^")) {
      pos++;
      return { kind: "binary", op: "^", left: base, right: unary() };
    }
    return base;
  };
  const unary = (): Node => {
    if (isOp("-", "not")) {
      const op = tokens[pos++].value as "-" | "not";
      return { kind: "unary", op, arg: unary() };
    }
    if (isOp("+")) {
      pos++;
      return unary();
    }
    return power();
  };
  const product = binary(unary, ["*", "/", "%"]);
  const sum = binary(product, ["+", "-"]);
  const compare = binary(sum, ["<", "<=", ">", ">=", "==", "!="]);
  const and = binary(compare, ["and"]);
  const or = binary(and, ["or"]);
  const node = or();
  if (pos < tokens.length) throw new Error(`unexpected "${tokens[pos].value}"`);
  return node;
}

/** Field names an expression uses. */
export function fieldsOf(node: Node, out = new Set<string>()): Set<string> {
  if (node.kind === "field") out.add(node.name);
  else if (node.kind === "unary") fieldsOf(node.arg, out);
  else if (node.kind === "binary") (fieldsOf(node.left, out), fieldsOf(node.right, out));
  else if (node.kind === "call") for (const a of node.args) fieldsOf(a, out);
  return out;
}

/** Evaluate per point; `fields` maps each field name used to its values. */
export function evaluate(node: Node, fields: Map<string, ArrayLike<number>>, count: number): Float32Array {
  const compile = (n: Node): ((i: number) => number) => {
    switch (n.kind) {
      case "number": {
        const v = n.value;
        return () => v;
      }
      case "field": {
        const values = fields.get(n.name);
        if (!values) throw new Error(`no field "${n.name}"`);
        return (i) => values[i];
      }
      case "unary": {
        const a = compile(n.arg);
        return n.op === "-" ? (i) => -a(i) : (i) => (a(i) ? 0 : 1);
      }
      case "call": {
        const fn = FUNCTIONS[n.name].fn;
        const args = n.args.map(compile);
        if (args.length === 1) return (i) => fn(args[0](i));
        return (i) => fn(...args.map((a) => a(i)));
      }
      case "binary": {
        const [l, r] = [compile(n.left), compile(n.right)];
        const ops: Record<string, (i: number) => number> = {
          "+": (i) => l(i) + r(i),
          "-": (i) => l(i) - r(i),
          "*": (i) => l(i) * r(i),
          "/": (i) => l(i) / r(i),
          "%": (i) => l(i) % r(i),
          "^": (i) => l(i) ** r(i),
          "<": (i) => +(l(i) < r(i)),
          "<=": (i) => +(l(i) <= r(i)),
          ">": (i) => +(l(i) > r(i)),
          ">=": (i) => +(l(i) >= r(i)),
          "==": (i) => +(l(i) === r(i)),
          "!=": (i) => +(l(i) !== r(i)),
          and: (i) => +(!!l(i) && !!r(i)),
          or: (i) => +(!!l(i) || !!r(i)),
        };
        return ops[n.op];
      }
    }
  };
  const f = compile(node);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = f(i);
  return out;
}
