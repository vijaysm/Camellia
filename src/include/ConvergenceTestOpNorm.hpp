//
//  ConvergenceTestSchwarz.hpp
//  Camellia
//
//  Created by Nate Roberts on 6/23/16.
//
//

#ifndef Camellia_ConvergenceTestOpNorm_hpp
#define Camellia_ConvergenceTestOpNorm_hpp

#include "BelosStatusTest.hpp"

namespace Camellia {
  template <class ScalarType, class MV, class OP>
  class ConvergenceTestOpNorm : public Belos::StatusTest<ScalarType,MV,OP>
  {
    Belos::StatusType _lastStatus;
    Teuchos::RCP<OP> _op;
    double _tol;
    
    public:
    //! @name Constructors/destructors
    //@{
    
    //! Constructor
    ConvergenceTestOpNorm(Teuchos::RCP<OP> op, double tol) {
      _lastStatus = Belos::Undefined;
      _op = op;
      _tol = tol;
    };
    
    //! Destructor
    virtual ~ConvergenceTestOpNorm() {};
    //@}
    
    //! @name Status methods
    //@{
    //! Check convergence status: Unconverged, Converged, Failed.
    /*! This method checks to see if the convergence criteria are met.  The calling routine may pass in the
     current native residual std::vector (the one naturally produced as part of the iterative method) or a
     pre-computed estimate of the two-norm of the current residual, or both or neither.  The calling routine
     should also indicate if the solution of the linear problem has been updated to be compatible with
     the residual.  Some methods, such as GMRES do not update the solution at each iteration.
     
     \return Belos::StatusType: Unconverged, Converged or Failed.
     */
    virtual Belos::StatusType checkStatus( Belos::Iteration<ScalarType,MV,OP>* iSolver )
    {
      std::vector<typename Teuchos::ScalarTraits<ScalarType>::magnitudeType> dummyNorms;
      
      Teuchos::RCP<const MV> residuals = iSolver->getNativeResiduals(&dummyNorms);
      
      MV Y(*residuals);
      _op->ApplyInverse(*residuals, Y);
      
      std::vector<double> myNorms(residuals->NumVectors());
      residuals->Dot(Y,&myNorms[0]);
      
      // we require all norm values to be less than tol.
      for (double value : myNorms)
      {
        if (value < 0)
        {
          std::cout << "WARNING: encountered negative norm value.  Perhaps Op was not positive definite?\n";
          return Belos::Undefined;
        }
        else if (value > _tol)
        {
          return Belos::Failed;
        }
      }
      // if we get here, then value < _tol for each
      return Belos::Passed;
    }
    
    //! Return the result of the most recent CheckStatus call.
    virtual Belos::StatusType getStatus() const
    {
      return _lastStatus;
    }
    //@}
    
    //! @name Reset methods
    //@{
    //! Informs the convergence test that it should reset its internal configuration to the initialized state.
    /*! This is necessary for the case when the status test is being reused by another solver or for another
     linear problem.  The status test may have information that pertains to a particular linear system.  The
     internal information will be reset back to the initialized state.  The user specified information that
     the convergence test uses will remain.
     */
    virtual void reset() {}
    //@}
    
    //! @name Print methods
    //@{
    
    //! Output formatted description of stopping test to output stream.
    virtual void print(std::ostream& os, int indent) const
    {
      // would be better to print something here!
    }
    
    //! Output the result of the most recent CheckStatus call.
    virtual void printStatus(std::ostream& os, Belos::StatusType type) const {
      os << std::left << std::setw(13) << std::setfill('.');
      switch (type) {
          case  Belos::Passed:
          os << "Passed";
          break;
          case  Belos::Failed:
          os << "Failed";
          break;
          case  Belos::Undefined:
          default:
          os << "**";
          break;
      }
      os << std::left << std::setfill(' ');
      return;
    };
    //@}
  };
}
#endif
