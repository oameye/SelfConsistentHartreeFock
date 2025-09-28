module best_practices
    using SelfConsistentHartreeFock
    using Aqua

    # Aqua.test_ambiguities([SelfConsistentHartreeFock]; broken=false)
    Aqua.test_all(SelfConsistentHartreeFock; ambiguities=false)
end
