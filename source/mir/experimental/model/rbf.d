module mir.experimental.model.rbf;

import std.traits : isFloatingPoint;
import std.experimental.allocator.mallocator : AlignedMallocator;

import ldc.attributes : fastmath;

import mir.ndslice.iteration : transposed;
import mir.ndslice.selection : diagonal;
import mir.internal.utility : isVector;
import mir.ndslice.slice;
import mir.glas.l3 : gemm;

//nothrow @nogc: will be with mir.glas inversion, scid is not nothrow @nogc

// Radial basis functions.
pure @fastmath
{
    /// Gaussian radial basis function.
    T gaussian(T)(T d, T r) if (isFloatingPoint!T)
    {
        import ldc.intrinsics : exp = llvm_exp;
        return exp(-d / (r * r));
    }
    /// Cauchy radial basis function.
    T cauchy(T)(T d, T r) if (isFloatingPoint!T)
    {
        return T(1.0) / (d / (r * r) + T(1.0));
    }
    /// Multiquadric radial basis function.
    T multiquadric(T)(T d, T r) if (isFloatingPoint!T)
    {
        import ldc.intrinsics : sqrt = llvm_sqrt;
        return sqrt((d / (r * r)) + T(1.0));
    }
    /// Inverse multiquadric radial basis function.
    T inverseMultiquadric(T)(T d, T r) if (isFloatingPoint!T)
    {
        import ldc.intrinsics : sqrt = llvm_sqrt;
        return T(1.0) / sqrt((d / (r * r)) + T(1.0));
    }
    /// Thin plate splines
    T thinPlates(T)(T d, T r) if (isFloatingPoint!T)
    {
        import ldc.intrinsics : log = llvm_log;
        return (d <= 0.0) ? 0.0 : r * r * log(r / d);
    }
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
)
in
{
    assert(data.length!1 == centers.length!1, "Input data and centers have to be of same dimension.");
    assert(design.shape == [centers.length, data.length], 
            "Design matrix size is invalid.");

    static if (isVector!Radii)
        assert(radii.length == center.length, "Radii count must be equal to the count of centers.");
}
body
{
    import mir.ndslice.iteration : transposed;
    static if (isVector!Radii)
    {
        static assert(isFloatingPoint!(ElementType!Radii),
                "Radius data have to be floating points.");
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
    for(; !data.empty; data.popFront, dt.popFront)
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
)
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

    for(; !h.empty; h.popFront, x.popFront)
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
)
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
)
in
{
    assert(weights.shape == values.shape);
}
body
{
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
)
in
{
    assert(weights.shape == values.shape);
    assert(lambdas.length == design.length!1);
}
body
{
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

    // Work buffers
    auto Abuf = makeSlice!T(AlignedMallocator.instance, [design.length!1, design.length!1]);
    auto Htybuf = makeSlice!T(AlignedMallocator.instance, [design.length!1, values.length!1]);

    scope(exit)
    {
        AlignedMallocator.instance.deallocate(Abuf.array);
        AlignedMallocator.instance.deallocate(Htybuf.array);
    }

    alias y = values;
    alias H = design;
    alias w = weights;
    auto Ht = H.transposed;
    auto A = Abuf.slice;
    auto Hty = Htybuf.slice;
};

// Invert matrix - wraps scid.linalg.invert
void invert(T)(Slice!(2, T*) matrix)
{
    import std.array : array;

    import scid.matrix;
    import scid.linalg : invert = invert;

    assert(matrix.length!0 == matrix.length!1);

    auto s = matrix.length!0;
    MatrixView!T v = MatrixView!T(matrix.ptr[0 .. s * s], s, s);

    invert(v);
}
