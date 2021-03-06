module mir.experimental.model.rbf;

import std.traits : isFloatingPoint;

import ldc.attributes : fastmath;

import mir.ndslice.slice : Slice, makeSlice, slice;
import mir.glas.l3 : gemm;
import mir.internal.utility : isVector;

nothrow @nogc:

// Radial basis functions.
pure @safe @fastmath
{
    /// Gaussian radial basis function.
    T gaussian(T = double)(T d, T r) if (isFloatingPoint!T)
    {
        import ldc.intrinsics : exp = llvm_exp;

        return exp(-d / (r * r));
    }
    /// Cauchy radial basis function.
    T cauchy(T = double)(T d, T r) if (isFloatingPoint!T)
    {
        return T(1.0) / (d / (r * r) + T(1.0));
    }
    /// Multiquadric radial basis function.
    T multiquadric(T = double)(T d, T r) if (isFloatingPoint!T)
    {
        import ldc.intrinsics : sqrt = llvm_sqrt;

        return sqrt((d / (r * r)) + T(1.0));
    }
    /// Inverse multiquadric radial basis function.
    T inverseMultiquadric(T = double)(T d, T r) if (isFloatingPoint!T)
    {
        import ldc.intrinsics : sqrt = llvm_sqrt;

        return T(1.0) / sqrt((d / (r * r)) + T(1.0));
    }
    /// Thin plate splines
    T thinPlates(T = double)(T d, T r) if (isFloatingPoint!T)
    {
        import ldc.intrinsics : log = llvm_log;

        return (d <= 0.0) ? 0.0 : r * r * log(r / d);
    }
}

// Check if given function has required radial basis function interface.
package(mir) template hasRbfInterface(alias fun)
{
    import std.traits : isSomeFunction, isFloatingPoint, ReturnType, allSatisfy, functionAttributes,
        FunctionAttribute, Parameters;

    alias FA = FunctionAttribute;

    enum hasRbfInterface = (isSomeFunction!fun && isFloatingPoint!(ReturnType!fun)
                && Parameters!fun.length == 2 && allSatisfy!(isFloatingPoint, Parameters!fun)
                && functionAttributes!fun & FA.pure_ && functionAttributes!fun & FA.safe);
}

nothrow @nogc @safe pure unittest
{
    import std.meta : AliasSeq;
    import std.traits : allSatisfy;

    alias RBFs = AliasSeq!
    (
        gaussian!float, cauchy!float, multiquadric!float, inverseMultiquadric!float, thinPlates!float,
        gaussian!double, cauchy!double, multiquadric!double, inverseMultiquadric!double, thinPlates!double,
        gaussian!real, cauchy!real, multiquadric!real, inverseMultiquadric!real, thinPlates!real
    );

    // should these be static asserts?
    assert(allSatisfy!(hasRbfInterface, RBFs));

    // non-RBFs
    void function() nonRbf1;
    float function(float) nonRbf2;
    int function(int) nonRbf3;
    void function(float, float) nonRbf4;

    assert(!hasRbfInterface!nonRbf1);
    assert(!hasRbfInterface!nonRbf2);
    assert(!hasRbfInterface!nonRbf3);
    assert(!hasRbfInterface!nonRbf4);
}

/++
Create the design matrix.

Params:
    centers = Center positions matrix (m-by-p).
    data = Training data (n-by-p).
    radii = Radius factors for radial basis function. Can be floating point
        scalar, as in same radius for all centers, or vector of floats, for
        radius per center. In vector case must be size of p.
    design = Output design matrix (m-by-n).
+/
void designRbf(alias rbf, T, Radii)
(
    Slice!(2, const(T)*) centers,
    Slice!(2, const(T)*) data,
    Radii radii,
    Slice!(2, T*) design
) if (isFloatingPoint!T && hasRbfInterface!rbf)
in
{
    assert(data.length!1 == centers.length!1, "Input data and centers have to be of same dimension.");
    assert(design.shape == [centers.length, data.length], "Design matrix size is invalid.");

    static if (isVector!Radii)
        assert(radii.length == center.length, "Radii count must be equal to the count of centers.");
}
body
{
    import mir.ndslice.iteration : transposed;

    static if (isVector!Radii)
    {
        static assert(isFloatingPoint!(ElementType!Radii), "Radius data have to be floating points.");
        auto r = radii.front;
        enum radius = `r.front`;
        enum popRadii = `r.popFront`;
    }
    else
    {
        static assert(isFloatingPoint!Radii, "Radius value has to be a floating point.");
        enum radius = `radii`;
        enum popRadii = ``;
    }

    auto dt = design.transposed; // column-wise iteration
    for (; !data.empty; data.popFront, dt.popFront)
    {
        designRbf!rbf(centers, data.front, mixin(radius), dt.front);
        mixin(popRadii);
    }
}

/++
Calculate single column of RBF design matrix.
+/
void designRbf(alias rbf, T, R)
(
    Slice!(2, const(T)*) center,
    Slice!(1, const(T)*) data,
    R radius,
    Slice!(1, T*) design
) if (isFloatingPoint!T && hasRbfInterface!rbf)
in
{
    assert(design.length == center.length);
}
body
{
    import ldc.intrinsics : sqrt = llvm_sqrt;

    alias h = design;
    alias x = center;
    alias c = data;

    immutable rr = radius * radius;
    immutable n = x.length!1;

    T r; // design result temp
    T d; // distance temp

    for (; !h.empty; h.popFront, x.popFront)
    {
        auto xr = x.front;
        r = T(0);
        for (size_t i = 0; !xr.empty; xr.popFront, ++i)
        {
            d = xr.front - c[i];
            r += d * d;
        }
        if (n > 1)
            r = sqrt(r);
        h.front = rbf(r, rr);
    }
}

/++
Most basic algorithm for weights calculation.

Params:
    values = Input values, used to build design matrix (m-by-n).
    design = Design matrix. (n-by-p, where p is point dimensionality of centers).
    weights = Output weight matrix (m-by-n);
+/
void basicWeights(T)
(
    Slice!(2, const(T)*) values,
    Slice!(2, const(T)*) design,
    Slice!(2, T*) weights
) if (isFloatingPoint!T)
in
{
    assert(weights.shape == values.shape);
}
body
{
    mixin(weightDataPrep);

    // w = inv(H' * H ) * H' * y;

    gemm(1.0, Ht, H, 0.0, A);
    invert(A);
    gemm(1.0, Ht, y, 0.0, Hty);
    gemm(1.0, A, Hty, 0.0, w);
}

/++
Calculate weights using global ridge regression method with given regularization value.

Params:
    values = Input values, used to build design matrix (m-by-n).
    design = Design matrix. (n-by-p, where p is point dimensionality of centers).
    lambda = Regularization parameter.
    weights = Output weight matrix (m-by-n);
+/

void ridgeGlobalWeights(T)
(
    Slice!(2, const(T)*) values,
    Slice!(2, const(T)*) design,
    T lambda,
    Slice!(2, T*) weights
) if (isFloatingPoint!T)
in
{
    assert(weights.shape == values.shape);
}
body
{
    import mir.ndslice.selection : diagonal;

    mixin(weightDataPrep);

    // w = inv(H' * H + lambda * eye(m)) * H' * y;

    gemm(1.0, Ht, H, 0.0, A);
    A.diagonal[] += lambda;
    invert(A);
    gemm(1.0, Ht, y, 0.0, Hty);
    gemm(1.0, A, Hty, 0.0, w);
}

/++
Calculate weights using local ridge regression method with given regularization values.

Params:
    values = Input values, used to build design matrix (m-by-n).
    design = Design matrix. (n-by-p, where p is point dimensionality of centers).
    lambdas = Regularization parameters. Vector of floats of size p.
    weights = Output weight matrix (m-by-n);
+/

void ridgeLocalWeights(T)
(
    Slice!(2, const(T)*) values,
    Slice!(2, const(T)*) design,
    Slice!(1, const(T)*) lambdas,
    Slice!(2, T*) weights
) if(isFloatingPoint!T)
in
{
    assert(weights.shape == values.shape);
    assert(lambdas.length == design.length!1);
}
body
{
    import mir.ndslice.selection : diagonal;

    mixin(weightDataPrep);

    // w = inv(H' * H + diag(lambdas)) * H' * y;

    gemm(1.0, Ht, H, 0.0, A);
    A.diagonal[] += lambdas[];
    invert(A);
    gemm(1.0, Ht, y, 0.0, Hty);
    gemm(1.0, A, Hty, 0.0, w);
}

private:

/+
Mixin string for weight calculation functions - allocates work
buffers, and aliases standard data members to known names.
+/
enum weightDataPrep = q{

    import std.experimental.allocator.mallocator : AlignedMallocator;
    import mir.ndslice.iteration : transposed;

    alias allocator = AlignedMallocator.instance;

    // Work buffers
    auto Abuf = makeSlice!T(allocator, [design.length!1, design.length!1]);
    auto Htybuf = makeSlice!T(allocator.instance, [design.length!1, values.length!1]);

    scope(exit)
    {
        allocator.instance.deallocate(Abuf.array);
        allocator.instance.deallocate(Htybuf.array);
    }

    alias y = values;
    alias H = design;
    alias w = weights;
    auto Ht = H.transposed;
    auto A = Abuf.slice;
    auto Hty = Htybuf.slice;
};

// lapack invert
void invert(T)(Slice!(2, T*) m) @nogc nothrow
in
{
    assert(m.length!0 == m.length!1, "invert: can only invert square matrices");
}
body
{
    import std.experimental.allocator.mallocator : Mallocator;

    alias allocator = Mallocator.instance;

    static if (is(T == float))
    {
        alias getrf = sgetrf_;
        alias getri = sgetri_;
    }
    else static if (is(T == double))
    {
        alias getrf = dgetrf_;
        alias getri = dgetri_;
    }
    else
    {
        static assert(0);
    }

    int l = cast(int)m.length;
    int info;
    int lwork = -1;

    T optimal;

    // Do workspace query
    getri(&l, null, &l, null, &optimal, &lwork, &info);

    lwork = cast(int)optimal;

    assert(lwork > 0);

    // Allocate workspace memory.
    int* ipiv = cast(int*)allocator.allocate(l * int.sizeof);
    T* work = cast(T*)allocator.allocate(lwork * T.sizeof);

    scope (exit)
    {
        allocator.deallocate(cast(void[])ipiv[0 .. l]);
        allocator.deallocate(cast(void[])work[0 .. lwork]);
    }

    getrf(&l, &l, m.ptr, &l, ipiv, &info);
    getri(&l, m.ptr, &l, ipiv, work, &lwork, &info);

    assert(info >= 0);
    return;
}

extern (C) nothrow @nogc:

void sgetrf_(int* m, int* n, float* a, int* lda, int* ipiv, int* info);
void dgetrf_(int* m, int* n, double* a, int* lda, int* ipiv, int* info);
void sgetri_(int* n, float* a, int* lda, int* ipiv, float* work, int* lwork, int* info);
void dgetri_(int* n, double* a, int* lda, int* ipiv, double* work, int* lwork, int* info);
