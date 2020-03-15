import React from 'react';
import styled from 'styled-components';
import Search from './search';

const Container = styled.div`
    width: 100vw;
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
`;

const Landing = () => {
    return (
        <Container>
            <Search />
        </Container>
    );
};

export default Landing;
