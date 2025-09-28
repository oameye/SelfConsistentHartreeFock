module code_linting
    using SelfConsistentHartreeFock
    using JET

    JET.test_package(SelfConsistentHartreeFock; target_defined_modules=true)
    rep = report_package("SelfConsistentHartreeFock")
end
