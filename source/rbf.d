module rbf;

import std.math : exp, sqrt;
import std.traits : isFloatingPoint;
import std.typecons : Tuple;

import mir.ndslice;
import mir.glas : GlasContext, gemm;

// ----------- Radial basis functions, where s = (x - c)^^2 / r^^2. ----------------------------

alias RadialBasisFunction(T) = T delegate(in T input);

/// Gaussian radial basis function.
T gaussian(T)(in T s) if (isFloatingPoint!T) { return exp(-s); };
/// Cauchy radial basis function.
T cauchy(T)(in T s) if (isFloatingPoint!T) { return T(1.0) / (s + T(1.0)); };
/// Multiquadric radial basis function.
T multiquadric(T)(in T s) if (isFloatingPoint!T) { return sqrt(s + T(1.0)); };
/// Inverse multiquadric radial basis function.
T inverseMultiquadric(T)(in T s) if (isFloatingPoint!T) { return T(1.0) / sqrt(s + T(1.0)); };

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
    alias TrainingData = Tuple!(Vector, "centers", Vector, "values");

    private
    {
        auto glas = new GlasContext(); // glas context used for matrix operations

        TrainingData _data; // training data.
        Vector _weights; // calculated training weights (hidden layer of the network).
        T _radius; // radius of a basis function.
        T _lambda; // regularization value for ridge regression.
    }

    @property
    {
        auto radius(in T value) {
            this._radius = value;
            return this;
        }

        auto lambda(in T value) {
            this._lambda = value;
            return this;
        }

        auto radius() const {
            return this._radius;
        }

        auto lambda() const {
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
        this._data.centers = x.slice;
        this._data.values = y.slice;

        auto H = design(x, this._data.centers);
        this._weights = calcWeights(H);

        return this;
    }

    /++
     Perform data fitting based on previous neural network training.
     +/
    Vector fit(Vector qx)
    {
        auto p = this._data.centers.length;
        auto pq = qx.length;
        auto Ht = design(qx, this._data.centers);
        auto ft = slice!double([pq, 1], 0.0);
        
        glas.gemm(ft, 1.0, Ht, this._weights.reshape(p, 1));

        return ft.reshape(qx.length);
    }

    private
    {
        // Create the RBF design matrix
        Matrix design(Vector x, Vector c)
        {
            auto p = x.length;
            auto m = c.length;
            auto rr = radius * radius;

            auto H = slice!T(p, m);
            auto d = slice!T(p);
            auto s = slice!T(p, p);

            foreach (j; 0 .. m)
            {
                s[] = 0.0;

                assumeSameStructure!("d", "x")(d, x).ndEach!(v => v.d = v.x - c[j], Yes.vectorized);
                auto dm = d.reshape(p, 1);

                glas.gemm(s, 1.0, dm, dm.transposed);
                auto sd = s.diagonal;

                auto h = H[0 .. $, j];
                foreach (i; 0 .. p)
                {
                    h[i] = rbf(sd[i] / rr);
                }
            }
            return H;
        }

        // Calculate weights using global ridge regression method with manualy determined regularization parameter.
        auto calcWeights(Slice!(2, T*) H)
        {
            auto yl = _data.values.length;
            auto A = slice!T([H.length!1, H.length!1], 0.0); // variance matrix
            auto w = slice!T([yl, 1], 0); // resulting weight values
            auto AHy = slice!T([yl, 1], 0);

            glas.gemm(A, 1.0, H.transposed, H);
            A.diagonal[] += lambda;
            A = invert(A);

            glas.gemm(AHy, 1.0, H.transposed, _data.values.reshape(yl, 1));
            glas.gemm(w, 1.0, A, AHy);

            return w.reshape(yl);
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
