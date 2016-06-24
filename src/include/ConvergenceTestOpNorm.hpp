//
//  ConvergenceTestSchwarz.hpp
//  Camellia
//
//  Created by Nate Roberts on 6/23/16.
//
//

#ifndef Camellia_ConvergenceTestSchwarz_hpp
#define Camellia_ConvergenceTestSchwarz_hpp

#include "BelosStatusTest.hpp"

namespace Camellia {
  template <class ScalarType, class MV, class OP>
  class ConvergenceTestSchwarz : public Belos::StatusTest<ScalarType,MV,OP>
  {
    Belos::StatusType _lastStatus;
    
    public:
    //! @name Constructors/destructors
    //@{
    
    //! Constructor
    StatusTest() {
      _lastStatus = Belos::Undefined;
    };
    
    //! Destructor
    virtual ~StatusTest() {};
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
    virtual Belos::StatusType checkStatus( Belos::Iteration<ScalarType,MV,OP>* iSolver ) = 0;
    
    //! Return the result of the most recent CheckStatus call.
    virtual Belos::StatusType getStatus() const = 0;
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
    virtual void print(std::ostream& os, int indent = 0) const = 0;
    
    //! Output the result of the most recent CheckStatus call.
    virtual void printStatus(std::ostream& os, Belos::StatusType type) const {
      os << std::left << std::setw(13) << std::setfill('.');
      switch (type) {
          case  Passed:
          os << "Passed";
          break;
          case  Failed:
          os << "Failed";
          break;
          case  Undefined:
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
