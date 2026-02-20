use crate::error::EntroGdError;

// A simple filter pipeline framework for processing data through a sequence of transformations.
// Each filter takes an input, processes it, and produces an output that can be fed into
// the next filter in the chain.
// Ill split most functions into a filter module, thus it might be useful to have functions constructing
// pipelines for common ideas here i will build a factory or just use simple helper functions
// with function deifinitions like this: fn create_processor() -> impl Filter<Input = i32, Output = String>

pub trait Filter {
    type Input;
    type Output;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError>;
}

pub struct Chain<A, B> {
    previous: A,
    next: B,
}

impl<A, B> Filter for Chain<A, B>
where
    A: Filter,
    B: Filter<Input = A::Output>,
{
    type Input = A::Input;
    type Output = B::Output;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let intermediate = self.previous.process(input)?;
        self.next.process(intermediate)
    }
}

pub trait FilterExt: Filter + Sized {
    fn then<Next>(self, next: Next) -> Chain<Self, Next>
    where
        Next: Filter<Input = Self::Output>,
    {
        Chain {
            previous: self,
            next,
        }
    }
}

impl<F: Filter> FilterExt for F {}
