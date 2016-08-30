module rbf;

import std.math : exp, sqrt;
import std.traits : isFloatingPoint;
import std.typecons : Tuple;

import mir.ndslice;
import mir.glas : GlasContext, gemm;

// ----------- Radial basis functions, where s = (x - c)^^2 / r^^2. ----------------------------

alias RadialBasisFunction(T) = T delegate(in T input);

/// Gaussian radial basis function.
T gaussian(T)(in T s) if (isFloatingPoint!T)
{
    return exp(-s);
}
/// Cauchy radial basis function.
T cauchy(T)(in T s) if (isFloatingPoint!T)
{
    return T(1.0) / (s + T(1.0));
}
/// Multiquadric radial basis function.
T multiquadric(T)(in T s) if (isFloatingPoint!T)
{
    return sqrt(s + T(1.0));
}
/// Inverse multiquadric radial basis function.
T inverseMultiquadric(T)(in T s) if (isFloatingPoint!T)
{
    return T(1.0) / sqrt(s + T(1.0));
}

// --------------------------------------------------------------------------------------------

/++
Radial basis function neural network model.

Single-layered, linear, non-parametric artificial neural network model.

Params:
    T = Value type used. Default is double.
    rbf = Radial basis function used. Default is gaussian.
+/
struct RbfNetwork(T = double, alias rbf = gaussian!T) if (isFloatingPoint!T)
{
    alias Vector = Slice!(1, T*);
    alias Matrix = Slice!(2, T*);
    alias TrainingData = Tuple!(Matrix, "centers", Matrix, "values");

    private
    {
        auto glas = new GlasContext(); // glas context used for matrix operations

        TrainingData _data; // training data.
        Matrix _weights; // calculated training weights (hidden layer of the network).
        T _radius; // radius of a basis function.
        T _lambda; // regularization value for ridge regression.
    }

    @property
    {
        auto radius(in T value)
        {
            this._radius = value;
            return this;
        }

        auto lambda(in T value)
        {
            this._lambda = value;
            return this;
        }

        auto radius() const
        {
            return this._radius;
        }

        auto lambda() const
        {
            return this._lambda;
        }
    }

    /++
     Train neural network with given data.

     Params:
         x = Center values, data node positions.
         y = Training value, outcome of trained function at x center position.
     +/
    auto train(Vector x, Vector y)
    {
        return train(x.reshape(x.length, 1), y.reshape(y.length, 1));
    }

    /// ditto
    auto train(Matrix x, Matrix y)
    {
        this._data.centers = x.slice;
        this._data.values = y.slice;

        auto H = design(x, x);
        this._weights = calcWeights(H);

        return this;
    }
    /++
     Perform data fitting based on previous neural network training.
     +/
    Vector fit(Vector qx)
    {
        auto ft = fit(qx.reshape(qx.length, 1));
        return ft.reshape(qx.length);
    }

    /// ditto
    Matrix fit(Matrix qx)
    in 
    {
        assert(qx.length!1 == _data.centers.length!1);
    }
    body
    {
        auto p = this._data.centers.length!0;
        auto d = this._data.values.length!1;
        auto pq = qx.length!0;

        auto Ht = design(qx, this._data.centers);
        auto ft = slice!T([pq, d], 0.0);

        glas.gemm(ft, 1.0, Ht, this._weights);

        return ft;
    }

    private
    {
        // Create the RBF design matrix
        Matrix design(Vector x, Vector c)
        {
            return design(x.reshape(x.length, 1), c.reshape(c.length, 1));
        }

        Matrix design(Matrix x, Matrix c)
        in
        {
            assert(x.length!1 == c.length!1);
        }
        body
        {
            immutable p = x.length;
            immutable m = c.length;
            auto H = slice!T(p, m);
            auto h = slice!T(p);

            foreach (j; 0 .. m)
            {
                designColumn(h, x, c[j]);
                H[0 .. $, j] = h[];
            }

            return H;
        }

        void designColumn(Vector h, Matrix x, Vector c)
        {
            immutable d = x.length!1;
            immutable p = x.length;
            immutable rr = radius * radius;

            foreach (i; 0 .. p)
            {
                auto v = T(0.0);
                foreach (ii; 0 .. d)
                {
                    auto dist = x[i][ii] - c[ii];
                    v += dist * dist;
                }
                if (d > 1)
                    v = sqrt(v);
                h[i] = rbf(v / rr);
            }
        }

        // Calculate weights using global ridge regression method
        auto calcWeights(Slice!(2, T*) H)
        {
            auto yl = _data.values.length!0;
            auto d = _data.values.length!1;
            auto A = slice!T([H.length!1, H.length!1], 0.0); // variance matrix
            auto w = slice!T([yl, d], 0); // resulting weight values
            auto AHy = slice!T([yl, d], 0);

            glas.gemm(A, 1.0, H.transposed, H);
            A.diagonal[] += lambda;
            A = invert(A);

            glas.gemm(AHy, 1.0, H.transposed, _data.values);
            glas.gemm(w, 1.0, A, AHy);

            return w;
        }

        // Invert matrix - wraps scid.linalg.invert
        auto invert(Slice!(2, T*) matrix)
        {
            import std.array : array;

            import scid.matrix;
            import scid.linalg : invert = invert;

            assert(matrix.length!0 == matrix.length!1);

            MatrixView!T v = MatrixView!T(matrix.byElement.array, matrix.length!0, matrix.length!1);
            invert(v);
            return v.array.sliced(matrix.shape);
        }
    }
}
